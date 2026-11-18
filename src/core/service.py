"""VerificationService: the single entry point behind Streamlit, FastAPI and MCP.

All three interfaces call these methods, so the same input yields the same
verdict and evidence ids everywhere. Heavy components (retriever, NLI verifier,
LLM client, NER) are injected, which keeps the service testable with fakes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.core.aliases import AliasResolver, ResolvedEntity, normalize_name
from src.core.config import Settings
from src.core.errors import (
    CTIError,
    EntityNotFoundError,
    InternalError,
    InvalidInputError,
    UnknownRelationError,
    VerifierUnavailableError,
)
from src.core.logging import trace_id_var
from src.core.models import (
    CONTENT_NOTICE,
    RELATIONS,
    ClaimSpec,
    ClaimVerification,
    Conflict,
    EntityInfo,
    EvidenceItem,
    IngestPreview,
    IngestResult,
    NeighborRef,
    NLIScores,
    QuestionPlan,
    SearchResponse,
    Timeline,
    TimelineEntry,
)
from src.core.planner import QuestionPlanner
from src.core.session_store import SessionIndex, SessionStore, text_fingerprint, validate_session_id
from src.core.uploads import extract_text_from_upload, validate_report_text
from src.verification.claim_aggregator import ClaimAggregator
from src.verification.context_resolver import ContextResolver

logger = logging.getLogger(__name__)

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RELATION_VERBS = {
    "USES": ("uses", "no longer uses"),
    "TARGETS": ("targets", "no longer targets"),
    "EXPLOITS": ("exploits", "no longer exploits"),
    "USES_TECHNIQUE": ("uses technique", "no longer uses technique"),
}
BIBLIO_RE = re.compile(r"(ftnref|references:\s*\[|sources:\s*\[|bibliography:\s*\[)", re.I)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
DEMO_PUBLISHER_PREFIX = "DEMO – SYNTHETIC | "


# --------------------------------------------------------------------------
# validation helpers (R5)
# --------------------------------------------------------------------------
def _check_text(value: Any, name: str, lo: int, hi: int) -> str:
    if not isinstance(value, str):
        raise InvalidInputError(f"{name} must be a string")
    v = " ".join(value.split())
    if not (lo <= len(v) <= hi):
        raise InvalidInputError(f"{name} must be {lo}-{hi} characters (got {len(v)})")
    if not any(ch.isalnum() for ch in v):
        raise InvalidInputError(f"{name} must contain letters or digits, not only symbols")
    if "\x00" in v:
        raise InvalidInputError(f"{name} contains NUL characters")
    return v


def _check_entity(value: Any, name: str) -> str:
    return _check_text(value, name, 1, 120)


def _check_iso_date(value: Optional[str], name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not ISO_DATE_RE.match(value):
        raise InvalidInputError(f"{name} must be an ISO date YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise InvalidInputError(f"{name} is not a real calendar date") from exc
    return value


def _check_top_k(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not (1 <= value <= 50):
        raise InvalidInputError("top_k must be an integer between 1 and 50")
    return value


def _check_relation(value: Any) -> str:
    if not isinstance(value, str) or value.upper() not in RELATIONS:
        raise UnknownRelationError(
            f"Unknown relation {value!r}. Valid relations: {', '.join(RELATIONS)}"
        )
    return value.upper()


def _looks_non_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 6:
        return False
    non_latin = sum(1 for c in letters if "LATIN" not in unicodedata.name(c, "LATIN"))
    return non_latin / len(letters) > 0.3


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return f"{prefix}{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _norm_text(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _publisher_key(pub: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (pub or "").lower()).strip() or "unknown"


# --------------------------------------------------------------------------
# internal candidate record
# --------------------------------------------------------------------------
@dataclass
class Cand:
    evidence_id: str
    text: str
    report_id: Optional[str]
    segment_id: Optional[str]
    publisher: Optional[str]
    pub_date: Optional[str]
    source_type: str = "corpus"
    evidence_type: str = "document_passage"
    scores: Dict[str, float] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    assertion_id: Optional[str] = None
    subject: Optional[str] = None
    object: Optional[str] = None
    predicate: Optional[str] = None
    polarity: Optional[str] = None
    trust_level: Optional[str] = None
    synthetic: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    nli: Optional[Dict[str, Any]] = None
    label: Optional[str] = None
    is_hedged: bool = False

    def sort_key(self) -> Tuple:
        best = max([self.scores.get(k, 0.0) for k in ("rrf_score", "score", "dense_score")] or [0.0])
        return (0 if self.evidence_type == "graph_assertion" else 1, -best, self.evidence_id)

    def to_item(self, max_chars: int) -> EvidenceItem:
        text = self.text if len(self.text) <= max_chars else self.text[: max_chars - 1] + "…"
        nli = None
        if self.nli is not None:
            nli = NLIScores(
                label=self.nli["label"],
                p_entailment=round(self.nli["p_entailment"], 4),
                p_contradiction=round(self.nli["p_contradiction"], 4),
                p_neutral=round(self.nli["p_neutral"], 4),
            )
        return EvidenceItem(
            evidence_id=self.evidence_id,
            report_id=self.report_id,
            segment_id=self.segment_id,
            publisher=(DEMO_PUBLISHER_PREFIX + (self.publisher or "")) if self.synthetic else self.publisher,
            pub_date=self.pub_date,
            undated=not self.pub_date,
            untrusted_text=text,
            source_type=self.source_type,  # type: ignore[arg-type]
            evidence_type=self.evidence_type,  # type: ignore[arg-type]
            retrieval_scores={k: round(float(v), 6) for k, v in sorted(self.scores.items())},
            retrieval_methods=sorted(self.methods),
            nli=nli,
            is_hedged=self.is_hedged,
            is_duplicate=self.is_duplicate,
            duplicate_of=self.duplicate_of,
            assertion_id=self.assertion_id,
            subject=self.subject,
            object=self.object,
            predicate=self.predicate,
            polarity=self.polarity,
            trust_level=self.trust_level,
            synthetic=self.synthetic,
        )


@dataclass
class Analysis:
    claim: ClaimSpec
    subject: ResolvedEntity
    object: ResolvedEntity
    candidates: List[Cand]
    gated: List[Cand]
    rejected_count: int
    duplicates: int
    excluded_undated: int
    excluded_by_date: int
    classified: List[Cand]
    warnings: List[str]
    steps: List[Dict[str, Any]]


class VerificationService:
    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Any,
        verifier: Any,
        aliases: AliasResolver,
        llm: Any = None,
        encode: Optional[Callable[[List[str]], Any]] = None,
        planner: Optional[QuestionPlanner] = None,
        store: Optional[SessionStore] = None,
        extractor: Any = None,
        clock: Callable[[], float] = time.time,
        overlay_docs: Optional[List[Dict[str, Any]]] = None,
    ):
        self.settings = settings
        self.retriever = retriever
        self.verifier = verifier
        self.aliases = aliases
        self.llm = llm
        self._encode = encode
        self._clock = clock
        self.store = store or SessionStore(
            ttl_secs=settings.session_ttl_secs,
            max_sessions=settings.max_sessions,
            confirm_ttl_secs=settings.confirm_token_ttl_secs,
            clock=clock,
        )
        self.planner = planner or QuestionPlanner(
            aliases,
            llm_chat=(self._llm_chat if llm is not None else None),
            llm_available=self._llm_available,
        )
        self._extractor = extractor
        self._model_lock = threading.RLock()
        self._verifications: Dict[str, ClaimVerification] = {}
        self._ver_lock = threading.Lock()
        self.overlay_loaded = 0
        if overlay_docs:
            self._attach_overlay(overlay_docs)
        self._graph_index: Optional[Dict[str, Any]] = None
        self.started_at = time.time()

    # ------------------------------------------------------------ factories
    @classmethod
    def from_settings(cls, settings: Settings) -> "VerificationService":
        from src.core.bootstrap import build_service

        return build_service(settings)

    # ---------------------------------------------------------------- LLM
    def _llm_available(self) -> bool:
        if self.llm is None or not self.settings.llm_enabled:
            return False
        try:
            return bool(self.llm.is_available())
        except Exception:
            return False

    def _llm_chat(self, system: str, user: str) -> Optional[str]:
        if not self._llm_available():
            return None
        attempts = 1 + max(0, self.settings.llm_retries)
        for _ in range(attempts):
            try:
                out = self.llm.chat_complete(system, user, temperature=0.0)
            except TypeError:
                out = self.llm.chat_complete(system, user)
            except Exception:
                out = None
            if out:
                return out
        return None

    def llm_status(self) -> Dict[str, Any]:
        if not self._llm_available():
            return {"connected": False, "model": None, "label": "LLM offline"}
        model = None
        try:
            model = self.llm.model_name()
        except Exception:
            model = getattr(self.llm, "model", None)
        return {"connected": True, "model": model, "label": f"LLM: {model}" if model else "LLM connected"}

    # ------------------------------------------------------------- overlay
    def _attach_overlay(self, docs: List[Dict[str, Any]]) -> None:
        """Demo overlay (H2): explicit, labelled, never in eval mode."""
        if self.settings.mode == "eval":
            raise InvalidInputError("The demo overlay is refused when mode=eval")
        dense = self.retriever.dense
        for d in docs:
            d = dict(d)
            d["synthetic"] = True
            dense.documents.append(d)
            dense.texts.append(d.get("clean_text") or d.get("text") or "")
        with self._model_lock:
            import numpy as np

            new_emb = np.asarray(self._enc([d.get("clean_text") or d.get("text") or "" for d in docs]), dtype=np.float32)
            dense.embeddings = np.vstack([dense.embeddings, new_emb])
            if hasattr(self.retriever, "hybrid") and hasattr(self.retriever.hybrid, "bm25"):
                from rank_bm25 import BM25Okapi
                from src.retrieval import _tokens

                self.retriever.hybrid.bm25 = BM25Okapi([_tokens(t) for t in dense.texts])
                self.retriever.hybrid.tokenized_docs = [_tokens(t) for t in dense.texts]
        self.overlay_loaded = len(docs)

    def _enc(self, texts: List[str]):
        if self._encode is not None:
            return self._encode(texts)
        return self.retriever.dense.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    # ------------------------------------------------------------ evidence
    def _cand_from_item(self, item: Dict[str, Any], method: str) -> Optional[Cand]:
        if method == "graph":
            text = item.get("evidence_sentence") or item.get("evidence_text") or ""
            if not text.strip():
                return None
            aid = item.get("assertion_id")
            seg = item.get("segment_id") or item.get("sentence_id")
            rid = item.get("report_id")
            return Cand(
                evidence_id=str(aid) if aid else self._passage_id(rid, seg, text),
                text=str(text),
                report_id=rid,
                segment_id=seg,
                publisher=item.get("source_vendor") or item.get("source"),
                pub_date=item.get("publication_date") or None,
                evidence_type="graph_assertion",
                methods=["graph"],
                assertion_id=str(aid) if aid else None,
                subject=item.get("subject") or item.get("subject_id"),
                object=item.get("object") or item.get("object_id"),
                predicate=item.get("predicate"),
                polarity=item.get("polarity", "positive"),
                trust_level=item.get("trust_level"),
            )
        text = item.get("text") or item.get("clean_text") or ""
        if not str(text).strip():
            return None
        rid, seg = item.get("report_id"), item.get("segment_id")
        scores = {k: float(item[k]) for k in ("score", "rrf_score", "dense_score", "bm25_score") if k in item}
        stype = item.get("source_type") or "corpus"
        return Cand(
            evidence_id=self._passage_id(rid, seg, text),
            text=str(text),
            report_id=rid,
            segment_id=seg,
            publisher=item.get("publisher") or item.get("source"),
            pub_date=item.get("publication_date") or None,
            source_type=stype,
            scores=scores,
            methods=[method],
            synthetic=bool(item.get("synthetic", False)),
        )

    @staticmethod
    def _passage_id(rid: Optional[str], seg: Optional[str], text: str) -> str:
        if seg:
            return seg if (rid and seg.startswith(str(rid))) or not rid else f"{rid}::{seg}"
        return f"text::{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"

    def _retrieve_pool(self, queries: Sequence[str], session_id: Optional[str], pool_k: int) -> List[Cand]:
        merged: Dict[Tuple, Cand] = {}

        def _add(c: Optional[Cand]) -> None:
            if c is None:
                return
            key = (c.evidence_type == "graph_assertion", c.evidence_id)
            if key in merged:
                old = merged[key]
                for k, v in c.scores.items():
                    old.scores[k] = max(old.scores.get(k, v), v)
                old.methods = sorted(set(old.methods) | set(c.methods))
            else:
                merged[key] = c

        for q in queries:
            with self._model_lock:
                raw = self.retriever.retrieve(q, top_k=pool_k)
            for method in ("dense", "hybrid"):
                for it in raw.get(method, []):
                    _add(self._cand_from_item(it, method))
            for it in raw.get("graph", []):
                _add(self._cand_from_item(it, "graph"))
        sess = self.store.get(session_id) if session_id else None
        if sess is not None and sess.index is not None:
            for q in queries:
                with sess.lock, self._model_lock:
                    hits = sess.index.search(q, top_k=pool_k)
                for it in hits:
                    _add(self._cand_from_item(it, "session"))
        return sorted(merged.values(), key=Cand.sort_key)

    def _mark_duplicates(self, cands: List[Cand]) -> int:
        """H9: duplicates by (report, segment), assertion id and normalised text."""
        seen: Dict[str, str] = {}
        dups = 0
        for c in sorted(cands, key=Cand.sort_key):
            keys = []
            if c.report_id and c.segment_id:
                keys.append(f"seg::{c.report_id}::{c.segment_id}")
            if c.assertion_id:
                keys.append(f"assertion::{c.assertion_id}")
            nt = _norm_text(c.text)
            if nt:
                keys.append(f"text::{nt}")
            parent = next((seen[k] for k in keys if k in seen), None)
            if parent is not None and parent != c.evidence_id:
                c.is_duplicate, c.duplicate_of = True, parent
                dups += 1
            else:
                for k in keys:
                    seen[k] = c.evidence_id
        return dups

    # ---------------------------------------------------------------- gate
    @staticmethod
    def _is_bibliography(text: str) -> bool:
        t = (text or "").lower()
        if BIBLIO_RE.search(t):
            return True
        if len(re.findall(r"https?://", t)) >= 2:
            return True
        return len(re.findall(r"\[.*?\]\s*\(https?://.*?\)", t)) >= 2

    def _passes_gate(self, text: str, subj: ResolvedEntity, obj: ResolvedEntity) -> bool:
        if self._is_bibliography(text):
            return False
        sentences = [s for s in SENT_SPLIT_RE.split(text or "") if s.strip()] or [text or ""]
        return any(self.aliases.present(s, subj) and self.aliases.present(s, obj) for s in sentences)

    # ------------------------------------------------------------- analyze
    def _hypothesis(self, claim: ClaimSpec) -> str:
        pos, neg = RELATION_VERBS[claim.relation]
        verb = neg if claim.polarity == "negative" else pos
        return f"{claim.subject} {verb} {claim.object}."

    def _classify(self, c: Cand, hypothesis: str) -> None:
        if self.verifier is None:
            raise VerifierUnavailableError("The NLI verifier is not loaded")
        try:
            with self._model_lock:
                pred = self.verifier.classify_pair(evidence=c.text, claim=hypothesis)
        except CTIError:
            raise
        except Exception as exc:
            raise VerifierUnavailableError(f"NLI verification failed: {exc}") from exc
        p_ent, p_con, p_neu = float(pred.p_entailment), float(pred.p_contradiction), float(pred.p_neutral)
        s = self.settings
        label = "neutral"
        if p_ent > s.entail_threshold and p_ent >= p_con:
            label = "entailment"
        elif p_con > s.contra_threshold:
            label = "contradiction"
        c.label = label
        c.nli = {"label": label, "p_entailment": p_ent, "p_contradiction": p_con, "p_neutral": p_neu}
        c.is_hedged = bool(getattr(pred, "is_hedged", False))

    def _analyze(
        self,
        subject: str,
        relation: str,
        obj: str,
        polarity: str,
        as_of_date: Optional[str],
        date_from: Optional[str],
        session_id: Optional[str],
    ) -> Analysis:
        steps: List[Dict[str, Any]] = []
        warnings: List[str] = []
        claim = ClaimSpec(subject=subject, relation=relation, object=obj, polarity=polarity)  # type: ignore[arg-type]
        subj_e, obj_e = self.aliases.resolve(subject), self.aliases.resolve(obj)
        steps.append(
            {
                "step": "resolve_entities",
                "subject": {"canonical": subj_e.canonical_name, "mitre_id": subj_e.mitre_id, "in_registry": subj_e.in_registry},
                "object": {"canonical": obj_e.canonical_name, "mitre_id": obj_e.mitre_id, "in_registry": obj_e.in_registry},
            }
        )
        if subj_e.ambiguous:
            warnings.append(f"Entity {subject!r} is ambiguous in the registry; using {subj_e.canonical_name}")
        if obj_e.ambiguous:
            warnings.append(f"Entity {obj!r} is ambiguous in the registry; using {obj_e.canonical_name}")

        queries = list(
            dict.fromkeys(
                [
                    f"{subject} {obj}",
                    f"{subject} {RELATION_VERBS[relation][0]} {obj}",
                    f"{subj_e.canonical_name} {obj_e.canonical_name}",
                ]
            )
        )
        pool = self._retrieve_pool(queries, session_id, self.settings.retrieval_pool_k)
        steps.append({"step": "retrieve", "queries": queries, "candidates": len(pool)})

        gated, rejected = [], 0
        for c in pool:
            if self._passes_gate(c.text, subj_e, obj_e):
                gated.append(c)
            else:
                rejected += 1
        steps.append({"step": "entity_gate", "passed": len(gated), "rejected": rejected})

        dups = self._mark_duplicates(gated)
        steps.append({"step": "dedupe", "duplicates": dups})

        excluded_undated = excluded_by_date = 0
        kept: List[Cand] = []
        for c in gated:
            if as_of_date or date_from:
                if not c.pub_date:
                    excluded_undated += 1
                    continue
                if as_of_date and c.pub_date > as_of_date:
                    excluded_by_date += 1
                    continue
                if date_from and c.pub_date < date_from:
                    excluded_by_date += 1
                    continue
            kept.append(c)
        if excluded_undated:
            warnings.append(f"{excluded_undated} undated passage(s) excluded because a date filter was requested")
        steps.append(
            {"step": "date_filter", "as_of_date": as_of_date, "date_from": date_from,
             "excluded_undated": excluded_undated, "excluded_by_date": excluded_by_date}
        )

        to_verify = [c for c in kept if not c.is_duplicate][: self.settings.max_verify_passages]
        if len(kept) - dups > len(to_verify):
            warnings.append(f"Only the top {self.settings.max_verify_passages} passages were verified")
        hyp = self._hypothesis(claim)
        for c in to_verify:
            self._classify(c, hyp)
        steps.append({"step": "nli", "hypothesis": hyp, "verified": len(to_verify)})
        return Analysis(
            claim=claim, subject=subj_e, object=obj_e, candidates=pool, gated=gated,
            rejected_count=rejected, duplicates=dups, excluded_undated=excluded_undated,
            excluded_by_date=excluded_by_date, classified=to_verify, warnings=warnings, steps=steps,
        )

    # -------------------------------------------------------------- verdict
    def _decide(self, an: Analysis, polarity: str) -> Dict[str, Any]:
        classified = an.classified
        agg = ClaimAggregator.aggregate(
            self._hypothesis(an.claim),
            [{"label": c.label} for c in classified],
        )
        supporting = [c for c in classified if c.label == "entailment"]
        contradicting = [c for c in classified if c.label == "contradiction"]
        neutral = [c for c in classified if c.label == "neutral"]
        status = agg.status

        corpus_support = [c for c in supporting if c.source_type == "corpus" and not c.synthetic]
        publishers = {_publisher_key(c.publisher) for c in corpus_support if c.pub_date or c.publisher}
        hedged_only = bool(supporting) and all(c.is_hedged for c in supporting)
        user_only = bool(supporting) and not corpus_support and all(c.source_type == "user_upload" for c in supporting)
        if supporting and contradicting:
            tier = "CONFLICTING"
        elif supporting:
            if user_only:
                tier = "SINGLE_SOURCE"
            elif corpus_support and len(publishers) >= 2:
                tier = "CORROBORATED"
            else:
                tier = "SINGLE_SOURCE"
        else:
            tier = "NONE"

        # Conflict resolution on non-duplicate classified evidence only
        records = [
            {
                "label": c.label, "source": c.publisher, "publication_date": c.pub_date,
                "evidence": c.text,
            }
            for c in classified
        ]
        conflict = Conflict()
        if supporting and contradicting:
            res = ContextResolver.resolve(
                self._hypothesis(an.claim), records,
                temporal_threshold_days=self.settings.temporal_threshold_days,
            )
            direction = {
                "CONTRA_LATER": "contradiction_later",
                "SUPP_LATER": "support_later",
                "CONCURRENT": "concurrent",
                "UNKNOWN": "unknown",
            }.get(res.temporal_relationship or "", None)
            conflict = Conflict(
                type=res.resolution_status, delta_days=res.temporal_delta_days,
                direction=direction, explanation=res.explanation,
            )
        return {
            "status": status, "tier": tier, "supporting": supporting, "contradicting": contradicting,
            "neutral": neutral, "conflict": conflict, "user_only": user_only, "hedged_only": hedged_only,
        }

    # --------------------------------------------------------------- brief
    def _template_brief(self, claim: ClaimSpec, d: Dict[str, Any], an: Analysis) -> str:
        lines = [
            f"Claim: {claim.subject} {claim.relation}{' (negated)' if claim.polarity == 'negative' else ''} {claim.object}",
            f"Status: {d['status']} | Support tier: {d['tier']}",
            f"Supporting passages: {len(d['supporting'])}, contradicting: {len(d['contradicting'])}, neutral: {len(d['neutral'])}.",
        ]
        if d["hedged_only"]:
            lines.append("All supporting passages use hedged language (suspected/alleged); treat as tentative.")
        if d["user_only"]:
            lines.append("Support comes only from user-supplied uploads; it is not independent corroboration.")
        if d["conflict"].type != "NO_CONFLICT":
            lines.append(f"Conflict: {d['conflict'].type} – {d['conflict'].explanation}")
        if any(c.synthetic for c in an.classified):
            lines.append("DEMO – SYNTHETIC evidence is present in this result.")
        for tag, group in (("Supporting", d["supporting"]), ("Contradicting", d["contradicting"])):
            for c in group[:3]:
                lines.append(f"- {tag} [{c.evidence_id}] ({c.publisher or 'unknown publisher'}, {c.pub_date or 'undated'})")
        return "\n".join(lines)

    _CITE_RE = re.compile(r"\[([^\[\]]+)\]")

    def _llm_summary(self, claim: ClaimSpec, d: Dict[str, Any], valid_ids: set, warnings: List[str]) -> Optional[str]:
        """H10: every sentence must cite an existing evidence id, else the summary is dropped."""
        if not self._llm_available() or not (d["supporting"] or d["contradicting"]):
            return None
        ev_lines = []
        for tag, group in (("SUPPORTS", d["supporting"][:3]), ("CONTRADICTS", d["contradicting"][:3])):
            for c in group:
                ev_lines.append(f"[{c.evidence_id}] ({tag}, {c.publisher}, {c.pub_date}) {c.text[:400]}")
        system = (
            "You are a CTI analyst. Write at most 3 short sentences about the claim, using ONLY the evidence given. "
            "Mention both supporting and contradicting evidence when both exist. "
            "End EVERY sentence with the id of the evidence it relies on in square brackets, e.g. [A000123]."
        )
        user = f"Claim: {claim.subject} {claim.relation} {claim.object}\nEvidence:\n" + "\n".join(ev_lines)
        out = self._llm_chat(system, user)
        if not out:
            warnings.append("LLM summary unavailable; template brief used")
            return None
        sentences = [s.strip() for s in re.split(r"(?<=[.!?\]])\s+", out.strip()) if s.strip()]
        if not sentences:
            return None
        for s in sentences:
            ids = [i.strip() for grp in self._CITE_RE.findall(s) for i in grp.split(",")]
            if not ids or any(i not in valid_ids for i in ids):
                warnings.append("LLM summary dropped: a sentence cited no valid evidence id")
                return None
        return " ".join(sentences)

    # ------------------------------------------------------------ public API
    def verify_claim(
        self,
        subject: str,
        relation: str,
        object: str,  # noqa: A002 - matches the MCP parameter name
        as_of_date: Optional[str] = None,
        session_id: Optional[str] = None,
        polarity: str = "positive",
        date_from: Optional[str] = None,
        use_llm_summary: bool = True,
    ) -> ClaimVerification:
        subject = _check_entity(subject, "subject")
        obj = _check_entity(object, "object")
        relation = _check_relation(relation)
        as_of_date = _check_iso_date(as_of_date, "as_of_date")
        date_from = _check_iso_date(date_from, "date_from")
        session_id = validate_session_id(session_id)
        if polarity not in ("positive", "negative"):
            raise InvalidInputError("polarity must be 'positive' or 'negative'")

        sess = self.store.get(session_id) if session_id else None
        upload_state = sorted(sess.reports.values()) if sess else []
        vid = _stable_id("v_", subject.lower(), relation, obj.lower(), polarity, as_of_date, date_from, session_id, upload_state, self.overlay_loaded)
        trace_id = vid.replace("v_", "t_", 1)
        trace_id_var.set(trace_id)
        warnings: List[str] = []

        if _looks_non_english(subject) or _looks_non_english(obj):
            warnings.append("English corpus: non-English input cannot be matched")
            result = self._empty_verification(vid, trace_id, subject, relation, obj, polarity, as_of_date, warnings)
            self._remember(result, {"steps": [{"step": "non_english_short_circuit"}]}, session_id)
            return result

        an = self._analyze(subject, relation, obj, polarity, as_of_date, date_from, session_id)
        d = self._decide(an, polarity)
        warnings.extend(an.warnings)
        if not self._llm_available():
            warnings.append("LLM offline")
        if d["hedged_only"]:
            warnings.append("All supporting passages use hedged language (suspected/alleged); treat as tentative")
        if d["user_only"]:
            warnings.append("Support is user-supplied only; not independent corroboration")
        if any(c.synthetic for c in an.classified):
            warnings.append("DEMO – SYNTHETIC evidence present")

        template = self._template_brief(an.claim, d, an)
        narrative, source = template, "template"
        if use_llm_summary:
            valid_ids = {c.evidence_id for c in d["supporting"] + d["contradicting"]}
            summary = self._llm_summary(an.claim, d, valid_ids, warnings)
            if summary:
                narrative, source = summary, "llm"

        mc = self.settings.max_passage_chars
        result = ClaimVerification(
            verification_id=vid,
            claim=an.claim,
            status=d["status"],
            support_tier=d["tier"],
            polarity=polarity,  # type: ignore[arg-type]
            supporting=[c.to_item(mc) for c in d["supporting"]],
            contradicting=[c.to_item(mc) for c in d["contradicting"]],
            neutral_count=len(d["neutral"]),
            duplicate_count=an.duplicates,
            excluded_undated=an.excluded_undated,
            conflict=d["conflict"],
            as_of_date_applied="publication_date_filter" if (as_of_date or date_from) else None,
            user_supplied_only=d["user_only"],
            hedged_only=d["hedged_only"],
            narrative=narrative,
            narrative_source=source,  # type: ignore[arg-type]
            warnings=list(dict.fromkeys(warnings)),
            trace_id=trace_id,
        )
        trace = {
            "verification_id": vid,
            "trace_id": trace_id,
            "claim": an.claim.model_dump(),
            "steps": an.steps,
            "evidence": [c.to_item(mc).model_dump() for c in an.classified],
            "status": result.status,
            "support_tier": result.support_tier,
            "warnings": result.warnings,
        }
        self._remember(result, trace, session_id)
        return result

    def _empty_verification(
        self, vid: str, trace_id: str, subject: str, relation: str, obj: str, polarity: str,
        as_of_date: Optional[str], warnings: List[str],
    ) -> ClaimVerification:
        return ClaimVerification(
            verification_id=vid,
            claim=ClaimSpec(subject=subject, relation=relation, object=obj, polarity=polarity),  # type: ignore[arg-type]
            status="INSUFFICIENT_EVIDENCE", support_tier="NONE", polarity=polarity,  # type: ignore[arg-type]
            as_of_date_applied="publication_date_filter" if as_of_date else None,
            warnings=warnings, trace_id=trace_id, narrative="No evidence available.",
        )

    def _remember(self, result: ClaimVerification, trace: Dict[str, Any], session_id: Optional[str]) -> None:
        with self._ver_lock:
            self._verifications[result.verification_id] = result
            if len(self._verifications) > 2000:
                for k in list(self._verifications)[:500]:
                    del self._verifications[k]
        self.store.store_trace(session_id, result.verification_id, trace)

    def get_verification(self, verification_id: str) -> Optional[ClaimVerification]:
        with self._ver_lock:
            return self._verifications.get(verification_id)

    def get_trace(self, verification_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        tr = self.store.get_trace(verification_id, session_id)
        if tr is None:
            raise InvalidInputError(f"No trace for verification_id {verification_id!r}", hint="Traces are kept in memory for the current server run.")
        return tr

    def verify_question(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        question = _check_text(question, "question", 3, 1000)
        session_id = validate_session_id(session_id)
        trace_id_var.set(_stable_id("t_", question.lower(), session_id))
        warnings: List[str] = []
        if _looks_non_english(question):
            warnings.append("English corpus: non-English input cannot be matched")
            plan = QuestionPlan(intent="GENERAL_EXPLORATION", sub_claims=[], planner="heuristic")
            return {"plan": plan.model_dump(), "results": [], "overall_status": "INSUFFICIENT_EVIDENCE",
                    "warnings": warnings, "content_notice": CONTENT_NOTICE}
        plan, pw = self.planner.plan(question)
        warnings.extend(pw)

        if not plan.sub_claims:
            related = []
            if plan.entities:
                try:
                    info = self.get_entity(plan.entities[0])
                    related = [n.model_dump() for n in info.neighbors]
                except CTIError:
                    pass
            return {
                "plan": plan.model_dump(), "results": [], "related_entities": related,
                "overall_status": "LOOKUP" if related else "INSUFFICIENT_EVIDENCE",
                "warnings": warnings, "content_notice": CONTENT_NOTICE,
            }

        ts = plan.temporal_scope or {}
        results = [
            self.verify_claim(
                sc.subject, sc.relation, sc.object, as_of_date=ts.get("date_to"),
                date_from=ts.get("date_from"), session_id=session_id, polarity=sc.polarity,
                use_llm_summary=False,
            )
            for sc in plan.sub_claims
        ]
        statuses = {r.status for r in results}
        if len(statuses) == 1:
            overall = statuses.pop()
        elif "CONFLICTING_EVIDENCE" in statuses or {"SUPPORTED", "CONTRADICTED"} <= statuses:
            overall = "CONFLICTING_EVIDENCE"
        elif "SUPPORTED" in statuses:
            overall = "SUPPORTED"
        elif "CONTRADICTED" in statuses:
            overall = "CONTRADICTED"
        else:
            overall = "INSUFFICIENT_EVIDENCE"
        return {
            "plan": plan.model_dump(),
            "results": [r.model_dump() for r in results],
            "overall_status": overall,
            "warnings": list(dict.fromkeys(warnings + [w for r in results for w in r.warnings])),
            "content_notice": CONTENT_NOTICE,
        }

    # ---------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        publishers: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> SearchResponse:
        query = _check_text(query, "query", 3, 1000)
        top_k = _check_top_k(top_k, self.settings.top_k)
        date_from = _check_iso_date(date_from, "date_from")
        date_to = _check_iso_date(date_to, "date_to")
        session_id = validate_session_id(session_id)
        if publishers is not None and (not isinstance(publishers, list) or any(not isinstance(p, str) for p in publishers)):
            raise InvalidInputError("publishers must be a list of strings")
        warnings: List[str] = []
        trace_id = _stable_id("t_", query.lower(), top_k, date_from, date_to, publishers, session_id)
        trace_id_var.set(trace_id)
        if _looks_non_english(query):
            warnings.append("English corpus: non-English input cannot be matched")
            return SearchResponse(query=query, results=[], total=0, warnings=warnings, trace_id=trace_id)

        pool = self._retrieve_pool([query], session_id, max(top_k * 3, self.settings.retrieval_pool_k))
        self._mark_duplicates(pool)
        pool = [c for c in pool if not c.is_duplicate]
        want = {_publisher_key(p) for p in publishers} if publishers else None
        out: List[Cand] = []
        undated = 0
        for c in pool:
            if want and _publisher_key(c.publisher) not in want:
                continue
            if date_from or date_to:
                if not c.pub_date:
                    undated += 1
                    continue
                if date_from and c.pub_date < date_from:
                    continue
                if date_to and c.pub_date > date_to:
                    continue
            out.append(c)
        if undated:
            warnings.append(f"{undated} undated passage(s) excluded because a date filter was requested")
        if not self._llm_available():
            pass  # search never needs the LLM
        total = len(out)
        items = [c.to_item(self.settings.max_passage_chars) for c in out[:top_k]]
        return SearchResponse(
            query=query, results=items, total=total, truncated=total > top_k,
            warnings=warnings, trace_id=trace_id,
        )

    # ---------------------------------------------------------- graph access
    def _graph(self) -> Dict[str, Any]:
        if self._graph_index is None:
            g = self.retriever.graph.graph
            nodes = {n["id"]: n for n in g.get("nodes", []) if n.get("id")}
            by_node: Dict[str, List[Dict[str, Any]]] = {}
            for link in g.get("links", []):
                by_node.setdefault(link["source"], []).append(link)
                by_node.setdefault(link["target"], []).append(link)
            self._graph_index = {"nodes": nodes, "links": by_node}
        return self._graph_index

    def _find_node(self, ent: ResolvedEntity) -> Optional[Dict[str, Any]]:
        gi = self._graph()
        lookup = self.retriever.graph.entity_lookup
        for term in sorted(ent.terms, key=lambda t: (-len(t), t)):
            nid = lookup.get(term)
            if nid and nid in gi["nodes"]:
                return gi["nodes"][nid]
        return None

    def get_entity(self, name: str) -> EntityInfo:
        name = _check_entity(name, "name")
        trace_id = _stable_id("t_", "entity", name.lower())
        ent = self.aliases.resolve(name)
        node = self._find_node(ent)
        if not ent.in_registry and node is None:
            raise EntityNotFoundError(f"No entity matching {name!r} in the registry or the corpus graph")
        neighbors: List[NeighborRef] = []
        if node is not None:
            gi = self._graph()
            agg: Dict[Tuple[str, str], int] = {}
            for link in gi["links"].get(node["id"], []):
                other_id = link["target"] if link["source"] == node["id"] else link["source"]
                other = gi["nodes"].get(other_id, {}).get("name") or other_id.split("::")[-1]
                rel = link.get("relation_type") or link.get("key") or "RELATED_TO"
                key = (rel if link["source"] == node["id"] else f"{rel} (inbound)", other)
                agg[key] = agg.get(key, 0) + int(link.get("evidence_count", 1) or 1)
            neighbors = [
                NeighborRef(relation=r, entity=e, evidence_count=n)
                for (r, e), n in sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
            ][:50]
        return EntityInfo(
            canonical_name=ent.canonical_name if ent.in_registry else (node or {}).get("name", ent.canonical_name),
            category=ent.category or (node or {}).get("category"),
            mitre_id=ent.mitre_id,
            aliases=list(ent.aliases),
            mention_count=int((node or {}).get("mention_count", 0)),
            first_seen=(node or {}).get("first_seen"),
            last_seen=(node or {}).get("last_seen"),
            neighbors=neighbors,
            in_registry=ent.in_registry,
            trace_id=trace_id,
        )

    # -------------------------------------------------------------- timeline
    def get_timeline(
        self, subject: str, object: str, relation: Optional[str] = None, session_id: Optional[str] = None  # noqa: A002
    ) -> Timeline:
        subject = _check_entity(subject, "subject")
        obj = _check_entity(object, "object")
        session_id = validate_session_id(session_id)
        warnings: List[str] = []
        if relation is None:
            relation = self._infer_relation(subject, obj)
            warnings.append(f"relation not given; assumed {relation}")
        else:
            relation = _check_relation(relation)
        an = self._analyze(subject, relation, obj, "positive", None, None, session_id)
        entries = []
        for c in an.classified:
            stance = {"entailment": "supports", "contradiction": "contradicts"}.get(c.label or "", "neutral")
            entries.append((c.pub_date or "9999-99-99", c.evidence_id, c, stance))
        entries.sort(key=lambda t: (t[0], t[1]))
        mc = self.settings.max_passage_chars
        return Timeline(
            subject=subject, object=obj, relation=relation,
            entries=[
                TimelineEntry(
                    pub_date=c.pub_date, publisher=c.publisher, stance=stance,  # type: ignore[arg-type]
                    evidence_id=c.evidence_id, untrusted_text=c.text[:mc],
                )
                for _, _, c, stance in entries
            ],
            warnings=warnings + an.warnings,
            trace_id=_stable_id("t_", "timeline", subject.lower(), obj.lower(), relation, session_id),
        )

    def _infer_relation(self, subject: str, obj: str) -> str:
        s, o = self.aliases.resolve(subject), self.aliases.resolve(obj)
        sn, on = self._find_node(s), self._find_node(o)
        if sn and on:
            for link in self._graph()["links"].get(sn["id"], []):
                if {link["source"], link["target"]} == {sn["id"], on["id"]}:
                    return link.get("relation_type") or link.get("key") or "USES"
        return "USES"

    # ---------------------------------------------------------------- ingest
    def _extract(self):
        if self._extractor is None:
            from src.core.extraction import ReportExtractor

            g = getattr(getattr(self.retriever, "graph", None), "graph", {}) or {}
            self._extractor = ReportExtractor(
                known_node_ids=[n.get("id") for n in g.get("nodes", [])],
                known_edges=[
                    (e["source"], e["target"], e.get("relation_type", e.get("key", ""))) for e in g.get("links", [])
                ],
            )
        return self._extractor

    def ingest_preview(
        self,
        text: str,
        title: str,
        publisher: str,
        pub_date: str,
        session_id: str,
        *,
        filename: Optional[str] = None,
        data: Optional[bytes] = None,
    ) -> IngestPreview:
        session_id = validate_session_id(session_id)
        if not session_id:
            raise InvalidInputError("session_id is required for uploads")
        title = _check_text(title, "title", 1, 200)
        publisher = _check_text(publisher, "publisher", 1, 120)
        pub_date = _check_iso_date(pub_date, "pub_date") or ""
        if not pub_date:
            raise InvalidInputError("pub_date is required (YYYY-MM-DD)")
        warnings: List[str] = []
        if data is not None:
            text, ew = extract_text_from_upload(
                data, filename or "upload.txt", self.settings.max_upload_bytes, self.settings.max_upload_pages
            )
            warnings.extend(ew)
        else:
            text = validate_report_text(text, self.settings.max_upload_bytes)

        sess = self.store.get(session_id, create=True)
        fp = text_fingerprint(text)
        with sess.lock:
            duplicate_of = sess.reports.get(fp)
        report_id = f"UPLOAD-{fp[:10].upper()}"
        extraction = self._extract().extract(text, report_id=report_id, publisher=publisher, pub_date=pub_date, title=title)
        warnings.extend(extraction.get("notes", []))
        if duplicate_of:
            warnings.append(f"This report was already ingested in this session as {duplicate_of}")
        if not extraction["segments"]:
            raise InvalidInputError("No sentences could be extracted from the report")
        pend = self.store.new_pending(
            sess, report_id=report_id, title=title, publisher=publisher, pub_date=pub_date,
            fingerprint=fp, segments=extraction["segments"], extraction=extraction,
        )
        preview_id = _stable_id("p_", report_id, session_id)
        return IngestPreview(
            preview_id=preview_id, confirm_token=pend.token, title=title, publisher=publisher,
            pub_date=pub_date, sentences=len(extraction["segments"]),
            entities=extraction["entities"][:100], relations=extraction["relations"][:100],
            warnings=warnings, duplicate_of=duplicate_of,
            trace_id=_stable_id("t_", "ingest", report_id, session_id),
        )

    def ingest_confirm(self, confirm_token: str, session_id: str) -> IngestResult:
        session_id = validate_session_id(session_id)
        if not session_id:
            raise InvalidInputError("session_id is required")
        if not isinstance(confirm_token, str) or not confirm_token:
            raise InvalidInputError("confirm_token is required")
        pend = self.store.take_pending(session_id, confirm_token)
        sess = self.store.get(session_id, create=True)
        with sess.lock:
            if pend.fingerprint in sess.reports:
                return IngestResult(
                    report_id=sess.reports[pend.fingerprint], segments_indexed=0, new_nodes=0, new_edges=0,
                    expires_at=self._iso(self.store.expires_at(sess)), duplicate=True,
                    trace_id=_stable_id("t_", "confirm", pend.report_id, session_id),
                )
            if sess.index is None:
                sess.index = SessionIndex(encode=self._enc_locked)
            sess.index.add(pend.segments)
            sess.reports[pend.fingerprint] = pend.report_id
            ex = pend.extraction
            new_nodes = [n for n in ex.get("graph_nodes", [])]
            new_edges = [e for e in ex.get("graph_edges", [])]
            sess.extra_nodes.extend(new_nodes)
            sess.extra_edges.extend(new_edges)
        return IngestResult(
            report_id=pend.report_id, segments_indexed=len(pend.segments),
            new_nodes=len(new_nodes), new_edges=len(new_edges),
            expires_at=self._iso(self.store.expires_at(sess)),
            trace_id=_stable_id("t_", "confirm", pend.report_id, session_id),
        )

    def _enc_locked(self, texts: List[str]):
        with self._model_lock:
            return self._enc(texts)

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------ STIX
    def export_stix(self, verification_ids: List[str]) -> Dict[str, Any]:
        from src.core.stix_export import build_bundle

        if not isinstance(verification_ids, list) or not verification_ids or len(verification_ids) > 50:
            raise InvalidInputError("verification_ids must be a list of 1-50 ids")
        items = []
        for vid in verification_ids:
            v = self.get_verification(vid)
            if v is None:
                raise InvalidInputError(f"Unknown verification_id {vid!r}", hint="Run verify_claim first; ids live for the current server run.")
            items.append(v)
        return build_bundle(items, self.aliases)

    # --------------------------------------------------------------- ops
    def stats(self) -> Dict[str, Any]:
        dense = self.retriever.dense
        graph = self.retriever.graph
        return {
            "reports": len({d.get("report_id") for d in dense.documents}),
            "segments": len(dense.documents),
            "graph_nodes": len(graph.graph.get("nodes", [])),
            "graph_edges": len(graph.graph.get("links", [])),
            "assertions": len(graph.assertions),
            "registry_entries": self.aliases.registry_size,
            "mode": self.settings.mode,
            "demo_overlay_records": self.overlay_loaded,
            "embedding_model": self.settings.embedding_model,
            "nli_model": self.settings.nli_model,
            "verifier_ready": self.verifier is not None,
            "llm": self.llm_status(),
            "active_sessions": self.store.active_count(),
            "manifest_sha256": getattr(self, "manifest_hash", None),
        }

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "uptime_secs": round(time.time() - self.started_at, 1)}

    def ready(self) -> Dict[str, Any]:
        problems = []
        if self.verifier is None:
            problems.append("NLI verifier not loaded")
        if self.retriever is None:
            problems.append("retriever not loaded")
        return {"ready": not problems, "problems": problems, "llm": self.llm_status()}
