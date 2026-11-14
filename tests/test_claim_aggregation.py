"""Unit tests for ClaimAggregator."""
import unittest
from src.verification.claim_aggregator import ClaimAggregator, ClaimAggregation
from src.verification.evidence_verifier import NLIPrediction


class TestClaimAggregation(unittest.TestCase):
    def _make_prediction(
        self,
        label: str,
        p_ent: float,
        p_con: float,
        p_neu: float,
        evidence: str,
        claim: str = "Emotet uses Trickbot.",
        assertion_id: str = "A001",
        source: str = "Zscaler",
        pub_date: str = "2021-01-29",
    ) -> NLIPrediction:
        return NLIPrediction(
            label=label,
            p_entailment=p_ent,
            p_contradiction=p_con,
            p_neutral=p_neu,
            evidence=evidence,
            claim=claim,
            evidence_id=f"ev::{assertion_id}",
            evidence_type="graph_assertion",
            assertion_id=assertion_id,
            report_id="REP-01",
            segment_id="SEG-01",
            source=source,
            publication_date=pub_date,
            confidence=0.95,
            trust_level="trusted",
        )

    def test_01_entailment_only_supported(self):
        p1 = self._make_prediction("entailment", 0.98, 0.01, 0.01, "Ev 1", assertion_id="A1")
        p2 = self._make_prediction("entailment", 0.95, 0.02, 0.03, "Ev 2", assertion_id="A2")

        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [p1, p2])
        self.assertEqual(agg.status, "SUPPORTED")
        self.assertEqual(agg.total_evidence, 2)
        self.assertEqual(agg.supporting_count, 2)
        self.assertEqual(agg.contradicting_count, 0)
        self.assertEqual(agg.neutral_count, 0)

    def test_02_contradiction_only_contradicted(self):
        p1 = self._make_prediction("contradiction", 0.01, 0.97, 0.02, "Ev 1", assertion_id="A1")
        p2 = self._make_prediction("contradiction", 0.02, 0.94, 0.04, "Ev 2", assertion_id="A2")

        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [p1, p2])
        self.assertEqual(agg.status, "CONTRADICTED")
        self.assertEqual(agg.total_evidence, 2)
        self.assertEqual(agg.supporting_count, 0)
        self.assertEqual(agg.contradicting_count, 2)
        self.assertEqual(agg.neutral_count, 0)

    def test_03_neutral_only_insufficient_evidence(self):
        p1 = self._make_prediction("neutral", 0.05, 0.05, 0.90, "Ev 1", assertion_id="A1")
        p2 = self._make_prediction("neutral", 0.10, 0.02, 0.88, "Ev 2", assertion_id="A2")

        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [p1, p2])
        self.assertEqual(agg.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(agg.total_evidence, 2)
        self.assertEqual(agg.supporting_count, 0)
        self.assertEqual(agg.contradicting_count, 0)
        self.assertEqual(agg.neutral_count, 2)

    def test_04_entailment_plus_contradiction_conflicting_evidence(self):
        p1 = self._make_prediction("entailment", 0.96, 0.02, 0.02, "Zscaler says Emotet dropped Trickbot", assertion_id="A1", source="Zscaler")
        p2 = self._make_prediction("contradiction", 0.01, 0.98, 0.01, "Intel 471 reports no Trickbot delivery", assertion_id="A2", source="Intel 471")

        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [p1, p2])
        self.assertEqual(agg.status, "CONFLICTING_EVIDENCE")
        self.assertEqual(agg.total_evidence, 2)
        self.assertEqual(agg.supporting_count, 1)
        self.assertEqual(agg.contradicting_count, 1)
        self.assertEqual(agg.neutral_count, 0)

    def test_05_entailment_plus_neutral_supported(self):
        p1 = self._make_prediction("entailment", 0.95, 0.01, 0.04, "Direct evidence", assertion_id="A1")
        p2 = self._make_prediction("neutral", 0.10, 0.05, 0.85, "Background info", assertion_id="A2")

        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [p1, p2])
        self.assertEqual(agg.status, "SUPPORTED")
        self.assertEqual(agg.total_evidence, 2)
        self.assertEqual(agg.supporting_count, 1)
        self.assertEqual(agg.contradicting_count, 0)
        self.assertEqual(agg.neutral_count, 1)

    def test_06_contradiction_plus_neutral_contradicted(self):
        p1 = self._make_prediction("contradiction", 0.01, 0.96, 0.03, "Refuting evidence", assertion_id="A1")
        p2 = self._make_prediction("neutral", 0.08, 0.04, 0.88, "Background info", assertion_id="A2")

        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [p1, p2])
        self.assertEqual(agg.status, "CONTRADICTED")
        self.assertEqual(agg.total_evidence, 2)
        self.assertEqual(agg.supporting_count, 0)
        self.assertEqual(agg.contradicting_count, 1)
        self.assertEqual(agg.neutral_count, 1)

    def test_07_empty_evidence_insufficient_evidence(self):
        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [])
        self.assertEqual(agg.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(agg.total_evidence, 0)
        self.assertEqual(agg.supporting_count, 0)
        self.assertEqual(agg.contradicting_count, 0)
        self.assertEqual(agg.neutral_count, 0)
        self.assertEqual(len(agg.evidence_results), 0)

    def test_08_invalid_label_raises_value_error(self):
        invalid_item = {"label": "MAYBE", "p_entailment": 0.5, "evidence": "text"}
        with self.assertRaises(ValueError):
            ClaimAggregator.aggregate("Emotet uses Trickbot.", [invalid_item])

    def test_09_provenance_and_probabilities_intact(self):
        pred = self._make_prediction(
            "entailment", 0.9912, 0.0034, 0.0054,
            evidence="Exact evidence sentence.",
            assertion_id="A000166",
            source="Zscaler",
            pub_date="2021-01-29"
        )
        agg = ClaimAggregator.aggregate("Emotet uses Trickbot.", [pred])
        rec = agg.evidence_results[0]

        self.assertEqual(rec["label"], "entailment")
        self.assertEqual(rec["p_entailment"], 0.9912)
        self.assertEqual(rec["p_contradiction"], 0.0034)
        self.assertEqual(rec["p_neutral"], 0.0054)
        self.assertEqual(rec["evidence"], "Exact evidence sentence.")
        self.assertEqual(rec["assertion_id"], "A000166")
        self.assertEqual(rec["source"], "Zscaler")
        self.assertEqual(rec["publication_date"], "2021-01-29")
        self.assertEqual(rec["confidence"], 0.95)
        self.assertEqual(rec["trust_level"], "trusted")


if __name__ == "__main__":
    unittest.main()
