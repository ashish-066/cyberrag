"""Claim-Level Evidence Aggregator.

Combines evidence-level NLI predictions into four CTI verification statuses:
- SUPPORTED: Supporting evidence only (or supporting + neutral).
- CONTRADICTED: Contradicting evidence only (or contradicting + neutral).
- CONFLICTING_EVIDENCE: Both supporting and contradicting evidence present across sources.
- INSUFFICIENT_EVIDENCE: Neutral evidence only (or no evidence available).

Preserves individual evidence predictions, probability scores, and source provenance.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence, Union


@dataclass
class ClaimAggregation:
    """Consolidated verification determination for a single factual claim."""
    claim: str
    status: str
    total_evidence: int
    supporting_count: int
    contradicting_count: int
    neutral_count: int
    evidence_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClaimAggregator:
    """Aggregates multiple NLI evidence predictions into a claim-level determination."""

    VALID_LABELS = {"entailment", "contradiction", "neutral"}

    @classmethod
    def aggregate(
        cls,
        claim: str,
        results: Sequence[Any],
    ) -> ClaimAggregation:
        """Aggregate evidence predictions for a single claim.

        Args:
            claim: The hypothesis statement being verified.
            results: Sequence of NLIPrediction instances or prediction dictionaries.

        Returns:
            ClaimAggregation containing consolidated status and full evidence records.

        Raises:
            ValueError: If claim is empty or any evidence item contains an invalid label.
        """
        if not claim or not str(claim).strip():
            raise ValueError("Claim cannot be empty")

        claim_clean = str(claim).strip()

        if not results:
            return ClaimAggregation(
                claim=claim_clean,
                status="INSUFFICIENT_EVIDENCE",
                total_evidence=0,
                supporting_count=0,
                contradicting_count=0,
                neutral_count=0,
                evidence_results=[],
            )

        supporting_count = 0
        contradicting_count = 0
        neutral_count = 0
        evidence_results: List[Dict[str, Any]] = []

        for item in results:
            if hasattr(item, "to_dict"):
                rec = item.to_dict()
                raw_label = getattr(item, "label", None)
            elif isinstance(item, dict):
                rec = dict(item)
                raw_label = item.get("label")
            else:
                raw_label = getattr(item, "label", None)
                rec = asdict(item) if hasattr(item, "__dataclass_fields__") else {"label": raw_label}

            if raw_label is None:
                raise ValueError("Evidence prediction item missing required 'label' field")

            lbl = str(raw_label).lower().strip()
            if lbl not in cls.VALID_LABELS:
                raise ValueError(
                    f"Invalid NLI label: '{raw_label}'. Expected one of: {sorted(cls.VALID_LABELS)}"
                )

            # Standardize label key in record
            rec["label"] = lbl
            evidence_results.append(rec)

            if lbl == "entailment":
                supporting_count += 1
            elif lbl == "contradiction":
                contradicting_count += 1
            elif lbl == "neutral":
                neutral_count += 1

        # Determine claim-level status
        if supporting_count > 0 and contradicting_count > 0:
            status = "CONFLICTING_EVIDENCE"
        elif supporting_count > 0:
            status = "SUPPORTED"
        elif contradicting_count > 0:
            status = "CONTRADICTED"
        else:
            status = "INSUFFICIENT_EVIDENCE"

        return ClaimAggregation(
            claim=claim_clean,
            status=status,
            total_evidence=len(evidence_results),
            supporting_count=supporting_count,
            contradicting_count=contradicting_count,
            neutral_count=neutral_count,
            evidence_results=evidence_results,
        )
