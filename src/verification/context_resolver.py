"""Context-Aware Contradiction Resolution Module for CTI.

Distinguishes NLI-level textual contradictions into actionable CTI determinations:
1. TEMPORAL_SUPERSEDENCE: Chronological evolution, infrastructure takedowns, or campaign shifts.
2. CAMPAIGN_DIVERGENCE: Different operational campaigns, targets, or payload variants.
3. GENUINE_ATTRIBUTION_DISPUTE: Direct factual disagreement between concurrent reporting vendors.
4. AMBIGUOUS_CONFLICT: Insufficient temporal metadata to establish chronology.
5. NO_CONFLICT: No contradicting evidence present.

Preserves research integrity: NLI classifies textual alignment; CTI context resolution
interprets multi-source intelligence context.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ContradictionResolution:
    """Detailed contextual determination for conflicting CTI evidence."""
    claim: str
    resolution_status: str          # TEMPORAL_SUPERSEDENCE, CAMPAIGN_DIVERGENCE, GENUINE_ATTRIBUTION_DISPUTE, AMBIGUOUS_CONFLICT, NO_CONFLICT
    is_conflict: bool
    explanation: str
    supporting_sources: List[str]
    contradicting_sources: List[str]
    supporting_dates: List[str]
    contradicting_dates: List[str]
    temporal_delta_days: Optional[int] = None
    temporal_relationship: Optional[str] = None  # CONTRA_LATER, SUPP_LATER, CONCURRENT, UNKNOWN
    resolvable: bool = True
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextResolver:
    """Resolves NLI-level conflicting evidence into structured CTI context categories."""

    DEFAULT_TEMPORAL_THRESHOLD_DAYS = 90

    CAMPAIGN_MARKERS = {
        "campaign", "operation", "variant", "phase", "wave",
        "sector", "financial", "healthcare", "defense", "energy", "government",
        "tag", "group", "actor", "target", "cluster"
    }

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date strings safely."""
        if not date_str:
            return None
        cleaned = str(date_str).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    @classmethod
    def resolve(
        cls,
        claim: str,
        evidence_results: Sequence[Dict[str, Any]],
        temporal_threshold_days: int = DEFAULT_TEMPORAL_THRESHOLD_DAYS,
    ) -> ContradictionResolution:
        """Analyze evidence for a claim and determine the root cause of any contradiction."""
        if not claim or not str(claim).strip():
            raise ValueError("Claim cannot be empty")

        claim_clean = str(claim).strip()
        supporting = [e for e in evidence_results if str(e.get("label", "")).lower() == "entailment"]
        contradicting = [e for e in evidence_results if str(e.get("label", "")).lower() == "contradiction"]

        # No conflict scenario
        if not supporting or not contradicting:
            return ContradictionResolution(
                claim=claim_clean,
                resolution_status="NO_CONFLICT",
                is_conflict=False,
                explanation="No conflicting evidence found across retrieved sources.",
                supporting_sources=sorted(list({e.get("source") for e in supporting if e.get("source")})),
                contradicting_sources=sorted(list({e.get("source") for e in contradicting if e.get("source")})),
                supporting_dates=sorted(list({e.get("publication_date") for e in supporting if e.get("publication_date")})),
                contradicting_dates=sorted(list({e.get("publication_date") for e in contradicting if e.get("publication_date")})),
                temporal_delta_days=None,
                temporal_relationship=None,
                resolvable=True,
                recommended_action="Proceed with aggregate NLI determination.",
            )

        # Extract sources and dates
        supp_sources = sorted(list({e.get("source") for e in supporting if e.get("source")}))
        contra_sources = sorted(list({e.get("source") for e in contradicting if e.get("source")}))

        supp_date_objs = [
            (cls._parse_date(e.get("publication_date")), e.get("publication_date"))
            for e in supporting if e.get("publication_date")
        ]
        contra_date_objs = [
            (cls._parse_date(e.get("publication_date")), e.get("publication_date"))
            for e in contradicting if e.get("publication_date")
        ]

        valid_supp_dates = [(dt, s) for dt, s in supp_date_objs if dt is not None]
        valid_contra_dates = [(dt, s) for dt, s in contra_date_objs if dt is not None]

        supp_dates_str = sorted(list({s for _, s in valid_supp_dates}))
        contra_dates_str = sorted(list({s for _, s in valid_contra_dates}))

        # If publication dates are missing for either side -> AMBIGUOUS_CONFLICT
        if not valid_supp_dates or not valid_contra_dates:
            return ContradictionResolution(
                claim=claim_clean,
                resolution_status="AMBIGUOUS_CONFLICT",
                is_conflict=True,
                explanation=(
                    "Evidence exhibits textual contradiction, but one or more sources lack "
                    "publication date metadata required to establish chronological ordering."
                ),
                supporting_sources=supp_sources,
                contradicting_sources=contra_sources,
                supporting_dates=supp_dates_str,
                contradicting_dates=contra_dates_str,
                temporal_delta_days=None,
                temporal_relationship="UNKNOWN",
                resolvable=False,
                recommended_action="Enrich evidence sources with publication dates before resolving conflict.",
            )

        earliest_supp_dt, earliest_supp_str = min(valid_supp_dates, key=lambda x: x[0])
        latest_supp_dt, latest_supp_str = max(valid_supp_dates, key=lambda x: x[0])
        earliest_contra_dt, earliest_contra_str = min(valid_contra_dates, key=lambda x: x[0])
        latest_contra_dt, latest_contra_str = max(valid_contra_dates, key=lambda x: x[0])

        # Check for temporal supersedence
        # Case A: Contradicting reports published significantly AFTER supporting reports
        if earliest_contra_dt > latest_supp_dt:
            delta_days = (earliest_contra_dt - latest_supp_dt).days
            if delta_days >= temporal_threshold_days:
                return ContradictionResolution(
                    claim=claim_clean,
                    resolution_status="TEMPORAL_SUPERSEDENCE",
                    is_conflict=True,
                    explanation=(
                        f"Contradicting evidence ({earliest_contra_str}) was published {delta_days} days "
                        f"after the latest supporting observation ({latest_supp_str}). Earlier relationship was "
                        "historically supported by reporting; newer evidence reflects subsequent infrastructure disruption, "
                        "operational transition, or campaign evolution over time."
                    ),
                    supporting_sources=supp_sources,
                    contradicting_sources=contra_sources,
                    supporting_dates=supp_dates_str,
                    contradicting_dates=contra_dates_str,
                    temporal_delta_days=delta_days,
                    temporal_relationship="CONTRA_LATER",
                    resolvable=True,
                    recommended_action=(
                        f"Preserve earlier relationship as historically supported prior to {earliest_contra_str}; "
                        "adopt newer contradicting evidence for ongoing operational threat assessments."
                    ),
                )

        # Case B: Supporting reports published significantly AFTER contradicting reports
        if earliest_supp_dt > latest_contra_dt:
            delta_days = (earliest_supp_dt - latest_contra_dt).days
            if delta_days >= temporal_threshold_days:
                return ContradictionResolution(
                    claim=claim_clean,
                    resolution_status="TEMPORAL_SUPERSEDENCE",
                    is_conflict=True,
                    explanation=(
                        f"Supporting evidence ({earliest_supp_str}) was published {delta_days} days "
                        f"after earlier contradicting/negative observations ({latest_contra_str}). "
                        "This indicates a newly emerged capability, resumed campaign, or updated vendor visibility."
                    ),
                    supporting_sources=supp_sources,
                    contradicting_sources=contra_sources,
                    supporting_dates=supp_dates_str,
                    contradicting_dates=contra_dates_str,
                    temporal_delta_days=delta_days,
                    temporal_relationship="SUPP_LATER",
                    resolvable=True,
                    recommended_action=(
                        f"Prioritize newer supporting evidence ({earliest_supp_str}) over outdated negative reports."
                    ),
                )

        # Check for campaign or sector divergence
        # Look for explicit campaign / target divergence keywords in evidence texts
        supp_text = " ".join(e.get("evidence", "") or e.get("evidence_text", "") for e in supporting).lower()
        contra_text = " ".join(e.get("evidence", "") or e.get("evidence_text", "") for e in contradicting).lower()


        # If evidence sets refer to distinct campaigns/clusters (e.g. tag differences)
        if ("tag" in supp_text or "campaign" in supp_text) and ("tag" in contra_text or "campaign" in contra_text):
            if any(term in supp_text for term in ("mor12", "mor13", "epoch", "cluster", "variant")) and \
               any(term in contra_text for term in ("epoch", "cluster", "variant", "campaign")):
                return ContradictionResolution(
                    claim=claim_clean,
                    resolution_status="CAMPAIGN_DIVERGENCE",
                    is_conflict=True,
                    explanation=(
                        "Supporting and contradicting passages describe distinct campaign clusters, operational tags, "
                        "or target environments, indicating scope divergence rather than direct factual contradiction."
                    ),
                    supporting_sources=supp_sources,
                    contradicting_sources=contra_sources,
                    supporting_dates=supp_dates_str,
                    contradicting_dates=contra_dates_str,
                    temporal_delta_days=abs((latest_supp_dt - earliest_contra_dt).days),
                    temporal_relationship="CONCURRENT",
                    resolvable=True,
                    recommended_action="Scope claim validity to the specific campaign tags or targets identified.",
                )

        # Case C: Concurrent reporting by distinct vendors -> GENUINE_ATTRIBUTION_DISPUTE
        delta_days = abs((latest_supp_dt - earliest_contra_dt).days)
        distinct_vendors = bool(set(supp_sources) ^ set(contra_sources)) or len(set(supp_sources + contra_sources)) > 1

        if distinct_vendors:
            vendors_str = f"'{', '.join(supp_sources)}' vs '{', '.join(contra_sources)}'"
            return ContradictionResolution(
                claim=claim_clean,
                resolution_status="GENUINE_ATTRIBUTION_DISPUTE",
                is_conflict=True,
                explanation=(
                    f"Direct attribution or capability dispute between reporting vendors ({vendors_str}) "
                    f"reporting concurrently (time delta: {delta_days} days). Neither report temporally supersedes the other."
                ),
                supporting_sources=supp_sources,
                contradicting_sources=contra_sources,
                supporting_dates=supp_dates_str,
                contradicting_dates=contra_dates_str,
                temporal_delta_days=delta_days,
                temporal_relationship="CONCURRENT",
                resolvable=False,
                recommended_action=(
                    "Retain both claims with vendor attribution; flag for CTI analyst review "
                    "to evaluate telemetry confidence and malware sample overlap."
                ),
            )

        # Same vendor conflicting concurrently -> AMBIGUOUS_CONFLICT
        return ContradictionResolution(
            claim=claim_clean,
            resolution_status="AMBIGUOUS_CONFLICT",
            is_conflict=True,
            explanation=(
                f"Conflicting observations from '{', '.join(supp_sources)}' within {delta_days} days "
                "without clear temporal supersedence or distinct campaign scoping."
            ),
            supporting_sources=supp_sources,
            contradicting_sources=contra_sources,
            supporting_dates=supp_dates_str,
            contradicting_dates=contra_dates_str,
            temporal_delta_days=delta_days,
            temporal_relationship="CONCURRENT",
            resolvable=False,
            recommended_action="Inspect full report text for contextual nuances or conditional statements.",
        )
