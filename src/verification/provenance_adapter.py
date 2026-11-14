"""Provenance Adapter for CTI Evidence Verification.

Normalizes Person 2's fused retrieval outputs (graph assertions and document passages)
into a common VerificationEvidence schema for NLI verification without modifying
or reinterpreting Person 2's retriever.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, List


@dataclass
class VerificationEvidence:
    """Normalized evidence record preserving full CTI provenance."""
    evidence_id: str
    evidence_type: str                  # "graph_assertion" or "document_passage"
    evidence_text: str
    claim: Optional[str] = None
    assertion_id: Optional[str] = None
    report_id: Optional[str] = None
    segment_id: Optional[str] = None
    source: Optional[str] = None
    publication_date: Optional[str] = None
    confidence: Optional[float] = None
    trust_level: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence record to dictionary preserving None fields."""
        return asdict(self)


class ProvenanceAdapter:
    """Adapts Person 2 retrieval outputs to the normalized VerificationEvidence schema."""

    @staticmethod
    def _first(item: Dict[str, Any], *keys: str) -> Optional[Any]:
        """Retrieve the first existing non-empty value for the provided keys."""
        for key in keys:
            if key in item and item[key] is not None:
                val = item[key]
                if isinstance(val, str) and not val.strip():
                    continue
                return val
        return None

    @classmethod
    def adapt_item(cls, item: Dict[str, Any], default_claim: Optional[str] = None) -> VerificationEvidence:
        """Convert a single Person 2 retrieval item to VerificationEvidence.

        Args:
            item: Raw dictionary from Person 2 (fused_results or retrieval_results).
            default_claim: Optional claim to attach if item does not define one.

        Returns:
            Normalized VerificationEvidence instance.

        Raises:
            ValueError: If evidence_text is missing or evidence type is unsupported.
        """
        raw_type = str(cls._first(item, "type", "evidence_type", "retrieval_method") or "").lower()

        if raw_type in ("graph", "graph_assertion"):
            evidence_type = "graph_assertion"
            evidence_text = cls._first(item, "evidence", "evidence_sentence", "evidence_text", "text")
            if not evidence_text:
                raise ValueError("Graph assertion item missing evidence text ('evidence' or 'evidence_sentence')")

            assertion_id = cls._first(item, "assertion_id", "id")
            evidence_id = str(assertion_id) if assertion_id else f"graph::{hash(evidence_text)}"

            claim = cls._first(item, "claim", "claim_text") or default_claim

            subject = cls._first(item, "subject")
            predicate = cls._first(item, "predicate", "relation_type")
            obj = cls._first(item, "object")
            if not claim and subject and predicate and obj:
                claim = f"{subject} {predicate} {obj}."

            # Parse confidence as float if present
            raw_conf = cls._first(item, "confidence")
            confidence = None
            if raw_conf is not None:
                try:
                    confidence = float(raw_conf)
                except (ValueError, TypeError):
                    confidence = None

            return VerificationEvidence(
                evidence_id=str(evidence_id),
                evidence_type=evidence_type,
                evidence_text=str(evidence_text),
                claim=str(claim) if claim else None,
                assertion_id=str(assertion_id) if assertion_id else None,
                report_id=cls._first(item, "report_id"),
                segment_id=cls._first(item, "segment_id", "sentence_id"),
                source=cls._first(item, "source", "source_vendor"),
                publication_date=cls._first(item, "publication_date"),
                confidence=confidence,
                trust_level=cls._first(item, "trust_level"),
                subject=str(subject) if subject else None,
                predicate=str(predicate) if predicate else None,
                object=str(obj) if obj else None,
            )

        elif raw_type in ("text", "document_passage", "dense", "hybrid"):
            evidence_type = "document_passage"
            evidence_text = cls._first(item, "text", "clean_text", "evidence_text", "evidence")
            if not evidence_text:
                raise ValueError("Document passage item missing evidence text ('text' or 'clean_text')")

            report_id = cls._first(item, "report_id")
            segment_id = cls._first(item, "segment_id", "sentence_id")
            if report_id and segment_id:
                evidence_id = f"{report_id}::{segment_id}"
            elif segment_id:
                evidence_id = str(segment_id)
            else:
                evidence_id = f"doc::{hash(evidence_text)}"

            claim = cls._first(item, "claim", "claim_text") or default_claim

            return VerificationEvidence(
                evidence_id=str(evidence_id),
                evidence_type=evidence_type,
                evidence_text=str(evidence_text),
                claim=str(claim) if claim else None,
                assertion_id=None,
                report_id=str(report_id) if report_id else None,
                segment_id=str(segment_id) if segment_id else None,
                source=cls._first(item, "source", "source_vendor"),
                publication_date=cls._first(item, "publication_date"),
                confidence=None,
                trust_level=None,
                subject=None,
                predicate=None,
                object=None,
            )
        else:
            raise ValueError(f"Unsupported evidence type: '{raw_type}'. Expected 'graph' or 'text'.")

    @classmethod
    def adapt_fused_results(
        cls,
        fused_results: List[Dict[str, Any]],
        default_claim: Optional[str] = None,
    ) -> List[VerificationEvidence]:
        """Convert a list of fused retrieval results to normalized VerificationEvidence."""
        return [cls.adapt_item(item, default_claim=default_claim) for item in fused_results]
