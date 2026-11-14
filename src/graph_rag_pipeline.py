"""End-to-end Graph-Augmented RAG orchestration with NLI Evidence Verification."""
from __future__ import annotations
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

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
            predictions = []
            for ev_item in flagged_items:
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
