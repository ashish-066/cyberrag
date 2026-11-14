"""End-to-end Graph-Augmented RAG orchestration with NLI Evidence Verification."""
from __future__ import annotations
import logging
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, Sequence

from src.hybrid_graph_rag import HybridGraphRAG
from src.evidence_fusion import EvidenceFusion
from src.context_builder import ContextBuilder
from src.answer_generator import AnswerGenerator
from src.verification.evidence_verifier import CTIEvidenceVerifier
from src.verification.provenance_adapter import ProvenanceAdapter, VerificationEvidence
from src.verification.claim_aggregator import ClaimAggregator
from src.verification.context_resolver import ContextResolver

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    def __init__(
        self,
        data_path="data/processed_segments.jsonl",
        graph_path="data/cti_knowledge_graph_trusted.json",
        assertions_path="data/assertions.jsonl",
        retriever=None,
        verifier=None,
    ):
        self.retriever = retriever or HybridGraphRAG(
            data_path=data_path,
            graph_path=graph_path,
            assertions_path=assertions_path,
        )
        self.fusion = EvidenceFusion()
        self.context_builder = ContextBuilder()
        self.answer_generator = AnswerGenerator()
        self.verifier = verifier
        self.provenance_adapter = ProvenanceAdapter()
        self.claim_aggregator = ClaimAggregator()
        self.context_resolver = ContextResolver()

    def _flag_duplicate_evidence(
        self,
        evidence_items: List[VerificationEvidence],
    ) -> List[VerificationEvidence]:
        """Identify duplicate evidence passages across graph assertions and document passages.

        Preserves both records but flags duplicates based on:
        - (report_id, segment_id)
        - assertion_id
        - Normalized evidence text

        Graph assertions are prioritized as primary evidence records; subsequent
        document passages covering the exact same segment or text are flagged as duplicates.
        """
        seen_keys: Dict[str, str] = {}

        def _get_item_keys(item: VerificationEvidence) -> List[str]:
            keys: List[str] = []
            if item.report_id and item.segment_id:
                keys.append(f"rep_seg::{item.report_id}::{item.segment_id}")
            if item.assertion_id:
                keys.append(f"assertion::{item.assertion_id}")
            norm_text = " ".join(item.evidence_text.split()).lower() if item.evidence_text else ""
            if norm_text:
                keys.append(f"text::{norm_text}")
            return keys

        # Pass 1: Register primary graph assertions
        for item in evidence_items:
            if item.evidence_type == "graph_assertion":
                keys = _get_item_keys(item)
                duplicate_parent = None
                for k in keys:
                    if k in seen_keys:
                        duplicate_parent = seen_keys[k]
                        break
                if duplicate_parent is not None and duplicate_parent != item.evidence_id:
                    item.is_duplicate = True
                    item.duplicate_of = duplicate_parent
                else:
                    item.is_duplicate = False
                    item.duplicate_of = None
                    for k in keys:
                        seen_keys[k] = item.evidence_id

        # Pass 2: Evaluate document passages against registered evidence
        for item in evidence_items:
            if item.evidence_type != "graph_assertion":
                keys = _get_item_keys(item)
                duplicate_parent = None
                for k in keys:
                    if k in seen_keys:
                        duplicate_parent = seen_keys[k]
                        break
                if duplicate_parent is not None and duplicate_parent != item.evidence_id:
                    item.is_duplicate = True
                    item.duplicate_of = duplicate_parent
                else:
                    item.is_duplicate = False
                    item.duplicate_of = None
                    for k in keys:
                        seen_keys[k] = item.evidence_id

        return evidence_items

    def _extract_subject_object(self, claim: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract subject and object strings from a relation claim."""
        claim_str = str(claim).strip()
        # Strictly match only the four known CTI relation types in capitals
        m = re.match(
            r"^\s*(.+?)\s+(?:USES|TARGETS|EXPLOITS|USES_TECHNIQUE)\s+(.+?)\.?\s*$",
            claim_str,
        )
        if m:
            subj = m.group(1).strip()
            obj = m.group(2).strip().rstrip(".")
            return subj, obj
        return None, None

    def _get_entity_terms(self, name: str) -> set[str]:
        """Get lowercase name and all known MITRE/CTI aliases for an entity."""
        terms = set()
        raw = (name or "").strip()
        if not raw:
            return terms

        clean = re.sub(
            r"^(?:ThreatActor|Malware|Tool|Organization|Location|Technique|Vulnerability)::",
            "",
            raw,
        ).strip()
        clean_lower = clean.lower()
        terms.add(clean_lower)
        terms.add(clean_lower.replace(" ", ""))
        terms.add(clean_lower.replace("-", ""))
        terms.add(clean_lower.replace("_", ""))

        try:
            from src.nlp.gazetteer import THREAT_ACTORS, MALWARE, TOOLS
            for registry in (THREAT_ACTORS, MALWARE, TOOLS):
                for canon, data in registry.items():
                    canon_clean = canon.lower()
                    aliases = [a.lower() for a in data.get("aliases", [])]
                    all_forms = {canon_clean, *aliases}
                    if clean_lower in all_forms:
                        terms.update(all_forms)
        except Exception:
            pass

        return {t for t in terms if len(t) >= 2}

    def _item_mentions_entity(self, item: Any, terms: set[str]) -> bool:
        """Check if an evidence item mentions at least one term from terms."""
        subj = getattr(item, "subject", "") or ""
        obj = getattr(item, "object", "") or ""
        for field in (subj, obj):
            f_clean = re.sub(r"^[A-Za-z]+::", "", str(field)).strip().lower()
            if f_clean and (f_clean in terms or any(t == f_clean for t in terms)):
                return True

        text = (
            getattr(item, "evidence_text", None)
            or getattr(item, "text", None)
            or getattr(item, "evidence", None)
            or (item.get("evidence_text") if isinstance(item, dict) else "")
            or (item.get("text") if isinstance(item, dict) else "")
            or ""
        ).lower()

        for t in terms:
            # Strictly require whole-word boundary matching to prevent substring leakage
            if re.search(r"\b" + re.escape(t) + r"\b", text):
                return True

        return False

    def _filter_evidence_for_claim(self, claim: str, items: Sequence[Any]) -> List[Any]:
        """Filter evidence items to only those that mention both the claim's subject and object."""
        subj, obj = self._extract_subject_object(claim)
        if not subj or not obj:
            return list(items)

        subj_terms = self._get_entity_terms(subj)
        obj_terms = self._get_entity_terms(obj)

        relevant = []
        for item in items:
            has_subj = self._item_mentions_entity(item, subj_terms)
            has_obj = self._item_mentions_entity(item, obj_terms)
            if has_subj and has_obj:
                relevant.append(item)

        return relevant

    def _verify_fused_results(
        self,
        fused: List[Dict[str, Any]],
        verification_claims: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Verify candidate propositions against retrieved evidence with NLI and aggregation."""
        # Obtain candidate propositions from graph assertions if not explicitly supplied
        if verification_claims is None:
            claims = list(dict.fromkeys(
                item.get("claim", "").strip()
                for item in fused
                if item.get("type") == "graph"
                and item.get("claim", "").strip()
            ))
        else:
            claims = list(dict.fromkeys(
                c.strip() for c in verification_claims if c.strip()
            ))

        if not claims:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "message": "No candidate claims were available.",
                "claims": [],
            }

        if self.verifier is None:
            logger.info("Initializing CTIEvidenceVerifier for GraphRAG verification...")
            self.verifier = CTIEvidenceVerifier()

        adapted_items = self.provenance_adapter.adapt_fused_results(fused)
        flagged_items = self._flag_duplicate_evidence(adapted_items)

        aggregated_claims = []
        for claim in claims:
            # Filter evidence items to only those that mention both the claim's subject and object (alias-aware)
            relevant_items = self._filter_evidence_for_claim(claim, flagged_items)

            if not relevant_items:
                agg = self.claim_aggregator.aggregate(claim, [])
                agg_dict = agg.to_dict()
                agg_dict["duplicate_count"] = 0
                agg_dict["unique_supporting_sources"] = []
                agg_dict["non_duplicate_supporting_count"] = 0
                agg_dict["resolution"] = {
                    "claim": claim,
                    "classification": "NO_CONFLICT",
                    "has_conflict": False,
                    "conflict_type": None,
                    "time_delta_days": None,
                    "explanation": "No relevant evidence mentioning both subject and object found in corpus.",
                    "supporting_sources": [],
                    "contradicting_sources": [],
                    "supporting_dates": [],
                    "contradicting_dates": [],
                    "recommended_action": "Treat claim as unverified / novel until primary reporting emerges.",
                }
                aggregated_claims.append(agg_dict)
                continue

            predictions = []
            for ev_item in relevant_items:
                pred = self.verifier.verify_evidence(ev_item, override_claim=claim)
                pred.is_duplicate = getattr(ev_item, "is_duplicate", False)
                pred.duplicate_of = getattr(ev_item, "duplicate_of", None)
                predictions.append(pred)

            agg = self.claim_aggregator.aggregate(claim, predictions)
            agg_dict = agg.to_dict()

            # Record duplicate and deduplication metrics
            agg_dict["duplicate_count"] = sum(
                1 for p in agg_dict["evidence_results"] if p.get("is_duplicate")
            )
            agg_dict["unique_supporting_sources"] = sorted(list({
                p["source"] for p in agg_dict["evidence_results"]
                if p.get("label") == "entailment" and p.get("source")
            }))
            agg_dict["non_duplicate_supporting_count"] = sum(
                1 for p in agg_dict["evidence_results"]
                if p.get("label") == "entailment" and not p.get("is_duplicate")
            )

            # CTI Context-Aware Contradiction Resolution
            resolution = self.context_resolver.resolve(claim, agg_dict["evidence_results"])
            agg_dict["resolution"] = resolution.to_dict()

            aggregated_claims.append(agg_dict)

        # Determine overall verification status across evaluated claims
        if len(aggregated_claims) == 1:
            overall_status = aggregated_claims[0]["status"]
        else:
            statuses = {c["status"] for c in aggregated_claims}
            if len(statuses) == 1:
                overall_status = statuses.pop()
            elif "CONFLICTING_EVIDENCE" in statuses:
                overall_status = "CONFLICTING_EVIDENCE"
            elif "SUPPORTED" in statuses and "CONTRADICTED" in statuses:
                overall_status = "CONFLICTING_EVIDENCE"
            elif "SUPPORTED" in statuses:
                overall_status = "SUPPORTED"
            elif "CONTRADICTED" in statuses:
                overall_status = "CONTRADICTED"
            else:
                overall_status = "INSUFFICIENT_EVIDENCE"

        return {
            "status": overall_status,
            "claims": aggregated_claims,
            "total_claims_evaluated": len(aggregated_claims),
        }

    def ask(
        self,
        query: str,
        top_k: int = 5,
        fused_top_k: int = 10,
        verify: bool = False,
        verification_claims: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Stage 1: Retrieval
        all_results = self.retriever.retrieve(query, top_k=top_k)

        # Stage 2: Evidence fusion
        fused = self.fusion.fuse(all_results, top_k=fused_top_k)

        # Stage 3: Context construction
        context = self.context_builder.build_context(fused)

        # Stage 4: Answer generation
        answer = self.answer_generator.generate_answer(query, fused)

        # Stage 5: Optional NLI verification
        verification = {
            "status": "NOT_RUN",
        }
        if verify:
            try:
                verification = self._verify_fused_results(
                    fused,
                    verification_claims=verification_claims,
                )
            except Exception as e:
                logger.error(f"Verification stage failed: {e}", exc_info=True)
                verification = {
                    "status": "ERROR",
                    "error": str(e),
                    "claims": [],
                }

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieval_results": all_results,
            "fused_results": fused,
            "verification": verification,
        }
