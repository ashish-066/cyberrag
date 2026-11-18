"""Query planning: heuristic planner built on NER + alias spans, optional LLM planner.

The heuristic planner is the default and the fallback. The LLM planner is only
accepted when its JSON passes schema validation and every entity is a verbatim
substring of the question (otherwise the heuristic result is used).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core.aliases import AliasResolver
from src.core.models import QuestionPlan, RELATIONS, SubClaim

logger = logging.getLogger(__name__)

# (regex, relation) in priority order. Matched against the text before an object.
RELATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:uses?|used|using)\s+technique\b|\bleverag(?:e|es|ed|ing)\s+technique\b", re.I), "USES_TECHNIQUE"),
    (re.compile(r"\bexploit(?:s|ed|ing)?\b", re.I), "EXPLOITS"),
    (re.compile(r"\b(?:target(?:s|ed|ing)?|attack(?:s|ed|ing)?|go(?:es)?\s+after|campaign(?:s)?\s+against)\b", re.I), "TARGETS"),
    (
        re.compile(
            r"\b(?:uses?|used|using|deploy(?:s|ed|ing)?|operat(?:e|es|ed|ing)|employ(?:s|ed|ing)?|deliver(?:s|ed|ing)?|"
            r"leverag(?:e|es|ed|ing)|rel(?:y|ies|ied)\s+on|associated\s+with|linked\s+to)\b",
            re.I,
        ),
        "USES",
    ),
]

NEGATION_RE = re.compile(
    r"\b(?:stop(?:ped|s|ping)?|no\s+longer|never|ceas(?:e|ed|es|ing)|discontinu(?:e|ed|es|ing)|"
    r"abandon(?:ed|s|ing)?|not|isn'?t|doesn'?t|didn'?t|hasn'?t|haven'?t)\b",
    re.I,
)
NOISE_WORDS = {
    "does", "did", "do", "is", "was", "are", "were", "has", "have", "had", "can", "could", "will",
    "still", "ever", "again", "also", "now", "currently", "actively", "really", "whether", "verify",
    "check", "true", "that", "the", "a", "an", "group", "actor", "malware",
}
WH_RE = re.compile(r"^\s*(?:what|which|who|whom|list|show)\b", re.I)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
SECTOR_WORDS = {
    "government", "banking", "banks", "hospitality", "healthcare", "energy", "finance", "financial",
    "telecom", "telecommunications", "education", "defense", "manufacturing", "retail", "transportation",
    "ukraine", "europe", "france", "germany", "united states", "us", "russia", "china", "iran", "israel",
}


@dataclass
class Span:
    start: int
    end: int
    text: str
    canonical: str
    category: Optional[str]


def _fake_sentence(text: str):
    from src.preprocessor.segmenter import PreprocessedSentence

    return PreprocessedSentence(
        sentence_id="q::p0::s0", paragraph_id="q::p0", report_id="q", source="query",
        publication_date="1970-01-01", paragraph_index=0, sentence_index=0,
        global_sentence_index=0, text=text,
    )


def make_default_ner() -> Optional[Callable[[str], List[Span]]]:
    """Gazetteer NER (no ML) from the existing engine; None if it cannot be built."""
    try:
        from src.nlp.ner_engine import CyberNEREngine

        engine = CyberNEREngine(enable_ml=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("NER engine unavailable for planning: %s", exc)
        return None

    def spans(text: str) -> List[Span]:
        out = []
        for ent in engine.extract_from_sentence(_fake_sentence(text)):
            out.append(Span(ent.start_char, ent.end_char, text[ent.start_char:ent.end_char], ent.canonical_name, ent.category))
        return out

    return spans


_AUTO = object()


class QuestionPlanner:
    def __init__(
        self,
        aliases: AliasResolver,
        ner: Any = _AUTO,
        llm_chat: Optional[Callable[[str, str], Optional[str]]] = None,
        llm_available: Optional[Callable[[], bool]] = None,
    ):
        self.aliases = aliases
        self.ner = make_default_ner() if ner is _AUTO else ner
        self.llm_chat = llm_chat
        self.llm_available = llm_available or (lambda: False)

    # --------------------------------------------------------------- public
    def plan(self, question: str) -> Tuple[QuestionPlan, List[str]]:
        """Return (plan, warnings). Never raises on bad LLM output."""
        warnings: List[str] = []
        if self.llm_chat and self.llm_available():
            plan = self._plan_llm(question, warnings)
            if plan is not None:
                return plan, warnings
        elif self.llm_chat is not None:
            warnings.append("LLM offline")
        return self._plan_heuristic(question), warnings

    # ------------------------------------------------------------ heuristic
    def _entity_spans(self, text: str) -> List[Span]:
        cands: List[Span] = []
        if self.ner:
            try:
                cands.extend(self.ner(text))
            except Exception as exc:  # pragma: no cover
                logger.warning("NER failed during planning: %s", exc)
        cands.extend(self._alias_spans(text))
        cands.extend(self._sector_spans(text))
        # longest-first, non-overlapping
        cands.sort(key=lambda s: (-(s.end - s.start), s.start))
        chosen: List[Span] = []
        for c in cands:
            if not any(c.start < o.end and o.start < c.end for o in chosen):
                chosen.append(c)
        if not chosen:
            chosen = self._capitalised_spans(text)
        chosen.sort(key=lambda s: s.start)
        return chosen

    def _alias_spans(self, text: str) -> List[Span]:
        out: List[Span] = []
        words = list(re.finditer(r"[A-Za-z0-9][\w.-]*", text))
        for n in range(4, 0, -1):
            for i in range(0, len(words) - n + 1):
                start, end = words[i].start(), words[i + n - 1].end()
                phrase = text[start:end]
                if len(phrase) < 3 or phrase.lower() in NOISE_WORDS:
                    continue
                ent = self.aliases.lookup(phrase)
                if ent is not None and not ent.ambiguous:
                    out.append(Span(start, end, phrase, ent.canonical_name, ent.category))
        return out

    def _sector_spans(self, text: str) -> List[Span]:
        out = []
        for word in sorted(SECTOR_WORDS, key=len, reverse=True):
            for m in re.finditer(r"\b" + re.escape(word) + r"\b", text, re.I):
                out.append(Span(m.start(), m.end(), m.group(0), m.group(0), "Organization"))
        return out

    def _capitalised_spans(self, text: str) -> List[Span]:
        out = []
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{2,}(?:\s+[A-Z][A-Za-z0-9_-]+)*\b", text):
            if m.group(0).lower() in NOISE_WORDS or m.start() == 0 and m.group(0).lower() in {"does", "did", "is", "was", "which", "what", "who", "has", "can"}:
                continue
            out.append(Span(m.start(), m.end(), m.group(0), m.group(0), None))
        return out

    @staticmethod
    def _relation_before(text: str, upto: int, floor: int) -> Optional[str]:
        """Nearest relation verb in text[floor:upto]."""
        seg = text[floor:upto]
        best: Tuple[int, Optional[str]] = (-1, None)
        for rx, rel in RELATION_PATTERNS:
            for m in rx.finditer(seg):
                if m.end() >= best[0]:
                    best = (m.end(), rel)
        return best[1]

    @staticmethod
    def _default_relation(span: Span) -> str:
        cat = (span.category or "").lower()
        if cat in ("organization", "location"):
            return "TARGETS"
        if cat == "cve" or span.text.upper().startswith("CVE-"):
            return "EXPLOITS"
        if cat == "technique":
            return "USES_TECHNIQUE"
        return "USES"

    def _extract_temporal(self, text: str) -> Optional[Dict[str, Optional[str]]]:
        years = [int(y) for y in YEAR_RE.findall(text)]
        if not years:
            return None
        low = text.lower()
        y = years[0]
        if len(years) >= 2:
            return {"date_from": f"{min(years)}-01-01", "date_to": f"{max(years)}-12-31"}
        if re.search(r"\b(?:after|since|following)\b", low):
            # 'after 2021' -> strictly later than the end of that year
            return {"date_from": f"{y + 1}-01-01" if "after" in low or "following" in low else f"{y}-01-01", "date_to": None}
        if re.search(r"\b(?:before|until|prior\s+to|by)\b", low):
            return {"date_from": None, "date_to": f"{y - 1}-12-31"}
        return {"date_from": f"{y}-01-01", "date_to": f"{y}-12-31"}

    def _plan_heuristic(self, question: str) -> QuestionPlan:
        q = question.strip()
        spans = self._entity_spans(q)
        temporal = self._extract_temporal(q)
        is_wh = bool(WH_RE.match(q))

        # Entities that are really years are not entities
        spans = [s for s in spans if not YEAR_RE.fullmatch(s.text)]
        entities = [s.text for s in spans]

        if is_wh and len(spans) <= 1:
            return QuestionPlan(
                intent="RELATIONAL_QUERY", sub_claims=[], temporal_scope=temporal,
                planner="heuristic", entities=entities,
            )
        if len(spans) < 2:
            return QuestionPlan(
                intent="GENERAL_EXPLORATION", sub_claims=[], temporal_scope=temporal,
                planner="heuristic", entities=entities,
            )

        subject = spans[0]
        sub_claims: List[SubClaim] = []
        prev_end = subject.end
        for obj in spans[1:]:
            # the clause is whatever sits between the previous entity and this one
            relation = self._relation_before(q, obj.start, prev_end) or self._relation_before(q, obj.start, 0) or self._default_relation(obj)
            clause_text = q[prev_end:obj.start]
            neg_scope = q[:obj.start] if not sub_claims else clause_text
            polarity = "negative" if NEGATION_RE.search(neg_scope) else "positive"
            if relation not in RELATIONS:
                relation = "USES"
            sub_claims.append(
                SubClaim(subject=subject.text, relation=relation, object=obj.text, polarity=polarity)
            )
            prev_end = obj.end

        intent = "FACTUAL_VERIFICATION"
        if re.search(r"\b(?:history|timeline|evolv\w*|over\s+time|when)\b", q, re.I):
            intent = "TEMPORAL_EVOLUTION"
        return QuestionPlan(
            intent=intent, sub_claims=sub_claims, temporal_scope=temporal,
            planner="heuristic", entities=entities,
        )

    # ------------------------------------------------------------------ LLM
    LLM_SYSTEM = (
        "You decompose a cyber threat intelligence question into JSON only:\n"
        '{"intent": "FACTUAL_VERIFICATION|RELATIONAL_QUERY|TEMPORAL_EVOLUTION|GENERAL_EXPLORATION",'
        ' "sub_claims": [{"subject": str, "relation": "USES|TARGETS|EXPLOITS|USES_TECHNIQUE",'
        ' "object": str, "polarity": "positive|negative"}],'
        ' "temporal_scope": {"date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null}|null}\n'
        "Rules: subject and object must be copied verbatim from the question. "
        "polarity is negative only for stopped / no longer / never statements. "
        "One sub_claim per (subject, object) pair."
    )

    def _plan_llm(self, question: str, warnings: List[str]) -> Optional[QuestionPlan]:
        try:
            raw = self.llm_chat(self.LLM_SYSTEM, f"Question: {question}")  # type: ignore[misc]
        except Exception as exc:
            warnings.append(f"LLM planner failed ({type(exc).__name__}); used heuristic planner")
            return None
        if not raw:
            warnings.append("LLM planner returned nothing; used heuristic planner")
            return None
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise ValueError("no JSON object")
            data = json.loads(m.group(0))
            claims = []
            for c in data.get("sub_claims", []):
                claims.append(
                    SubClaim(
                        subject=str(c["subject"]).strip(), relation=c["relation"],
                        object=str(c["object"]).strip(), polarity=c.get("polarity", "positive"),
                    )
                )
            ql = question.lower()
            for c in claims:
                if c.subject.lower() not in ql or c.object.lower() not in ql:
                    raise ValueError("entity not verbatim in question")
            ts = data.get("temporal_scope")
            if ts is not None and not isinstance(ts, dict):
                raise ValueError("temporal_scope must be an object")
            plan = QuestionPlan(
                intent=str(data.get("intent", "FACTUAL_VERIFICATION")),
                sub_claims=claims, temporal_scope=ts, planner="llm",
                entities=sorted({c.subject for c in claims} | {c.object for c in claims}),
            )
            if not claims:
                raise ValueError("no sub_claims")
            return plan
        except Exception as exc:
            warnings.append(f"LLM planner output rejected ({exc}); used heuristic planner")
            return None
