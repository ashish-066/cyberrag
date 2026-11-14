"""Evidence-claim NLI verification module using cross-encoder architecture.

Implements Level 1 NLI verification:
- Premise: Retrieved evidence sentence or passage.
- Hypothesis: Extracted relationship claim.
- Output: Softmax distribution over entailment, contradiction, and neutral.

Note on scientific integrity:
Probabilities returned are raw neural model softmax scores, not calibrated
statistical confidence intervals. NLI alone indicates textual alignment,
not ground-truth historical fact or multi-report conflict resolution.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from transformers import pipeline

logger = logging.getLogger(__name__)


@dataclass
class NLIPrediction:
    """Structured NLI verification result for a single (evidence, claim) pair with full provenance."""
    label: str                  # "entailment", "contradiction", or "neutral"
    p_entailment: float         # Probability score for entailment
    p_contradiction: float      # Probability score for contradiction
    p_neutral: float            # Probability score for neutral
    evidence: str               # Premise
    claim: str                  # Hypothesis

    # Provenance fields carried over from VerificationEvidence
    evidence_id: Optional[str] = None
    evidence_type: Optional[str] = None
    assertion_id: Optional[str] = None
    report_id: Optional[str] = None
    segment_id: Optional[str] = None
    source: Optional[str] = None
    publication_date: Optional[str] = None
    confidence: Optional[float] = None
    trust_level: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "p_entailment": round(self.p_entailment, 4),
            "p_contradiction": round(self.p_contradiction, 4),
            "p_neutral": round(self.p_neutral, 4),
            "evidence": self.evidence,
            "claim": self.claim,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "assertion_id": self.assertion_id,
            "report_id": self.report_id,
            "segment_id": self.segment_id,
            "source": self.source,
            "publication_date": self.publication_date,
            "confidence": self.confidence,
            "trust_level": self.trust_level,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
        }


class CTIEvidenceVerifier:
    """NLI cross-encoder verifier evaluating factual alignment between evidence and claims."""

    DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: int = -1,
    ):
        """Initialize NLI pipeline and validate label configuration."""
        self.model_name = model_name
        logger.info(f"Loading NLI verifier from {model_name}...")
        self.verifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=device,
        )

        # Validate label mapping from model config
        config = self.verifier.model.config
        id2label = getattr(config, "id2label", {})
        self.known_labels = {str(lbl).lower() for lbl in id2label.values()}
        logger.info(f"Loaded NLI model '{model_name}' with labels: {self.known_labels}")

        expected = {"entailment", "contradiction", "neutral"}
        if not expected.issubset(self.known_labels):
            logger.warning(
                f"Model id2label {id2label} does not contain all expected NLI classes {expected}. "
                f"Found: {self.known_labels}"
            )

    def _normalize_pipeline_output(self, raw_output: Any) -> Dict[str, float]:
        """Normalize transformers pipeline output into a label->score dictionary."""
        # Single pair returns list of dicts: [{'label': 'entailment', 'score': 0.99}, ...]
        # Batch returns list of lists: [[{'label': ...}], ...]
        items = raw_output[0] if (isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list)) else raw_output
        scores: Dict[str, float] = {}
        for entry in items:
            lbl = str(entry.get("label", "")).lower()
            val = float(entry.get("score", 0.0))
            scores[lbl] = val
        return scores

    def classify_pair(self, evidence: str, claim: str) -> NLIPrediction:
        """Classify a single (evidence, claim) pair using dual-sequence encoding.

        Args:
            evidence: The retrieved CTI evidence sentence/passage (Premise).
            claim: The asserted relationship or fact (Hypothesis).

        Returns:
            NLIPrediction with top label and probabilities.
        """
        ev_clean = str(evidence or "").strip()
        cl_clean = str(claim or "").strip()

        if not ev_clean or not cl_clean:
            return NLIPrediction(
                label="neutral",
                p_entailment=0.0,
                p_contradiction=0.0,
                p_neutral=1.0,
                evidence=ev_clean,
                claim=cl_clean,
            )

        # Native dual-sequence tokenization: text = Premise, text_pair = Hypothesis
        raw = self.verifier({"text": ev_clean, "text_pair": cl_clean})
        scores = self._normalize_pipeline_output(raw)

        p_ent = scores.get("entailment", 0.0)
        p_con = scores.get("contradiction", 0.0)
        p_neu = scores.get("neutral", 0.0)
        top_lbl = max(scores, key=scores.get) if scores else "neutral"

        return NLIPrediction(
            label=top_lbl,
            p_entailment=p_ent,
            p_contradiction=p_con,
            p_neutral=p_neu,
            evidence=ev_clean,
            claim=cl_clean,
        )

    def classify_batch(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 16,
    ) -> List[NLIPrediction]:
        """Classify a sequence of (evidence, claim) pairs in batches."""
        if not pairs:
            return []

        inputs = [{"text": str(e).strip(), "text_pair": str(c).strip()} for e, c in pairs]
        raw_outputs = self.verifier(inputs, batch_size=batch_size)

        predictions: List[NLIPrediction] = []
        for (e, c), raw in zip(pairs, raw_outputs):
            scores = self._normalize_pipeline_output(raw)
            p_ent = scores.get("entailment", 0.0)
            p_con = scores.get("contradiction", 0.0)
            p_neu = scores.get("neutral", 0.0)
            top_lbl = max(scores, key=scores.get) if scores else "neutral"

            predictions.append(
                NLIPrediction(
                    label=top_lbl,
                    p_entailment=p_ent,
                    p_contradiction=p_con,
                    p_neutral=p_neu,
                    evidence=str(e),
                    claim=str(c),
                )
            )
        return predictions

    def verify_evidence(
        self,
        item: Any,
        override_claim: Optional[str] = None,
    ) -> NLIPrediction:
        """Verify a VerificationEvidence record or dict, preserving all CTI provenance."""
        if hasattr(item, "evidence_text"):
            ev_text = item.evidence_text
            claim_text = override_claim or item.claim or ""
            ev_id = item.evidence_id
            ev_type = item.evidence_type
            aid = item.assertion_id
            rid = item.report_id
            sid = item.segment_id
            src = item.source
            pdate = item.publication_date
            conf = item.confidence
            tlevel = item.trust_level
        else:
            ev_text = item.get("evidence_text") or item.get("evidence") or item.get("text", "")
            claim_text = override_claim or item.get("claim", "")
            ev_id = item.get("evidence_id")
            ev_type = item.get("evidence_type")
            aid = item.get("assertion_id")
            rid = item.get("report_id")
            sid = item.get("segment_id")
            src = item.get("source")
            pdate = item.get("publication_date")
            conf = item.get("confidence")
            tlevel = item.get("trust_level")

        pred = self.classify_pair(ev_text, claim_text)
        pred.evidence_id = ev_id
        pred.evidence_type = ev_type
        pred.assertion_id = aid
        pred.report_id = rid
        pred.segment_id = sid
        pred.source = src
        pred.publication_date = pdate
        pred.confidence = conf
        pred.trust_level = tlevel
        pred.is_duplicate = getattr(item, "is_duplicate", False) if hasattr(item, "is_duplicate") else (item.get("is_duplicate", False) if isinstance(item, dict) else False)
        pred.duplicate_of = getattr(item, "duplicate_of", None) if hasattr(item, "duplicate_of") else (item.get("duplicate_of", None) if isinstance(item, dict) else None)
        return pred
