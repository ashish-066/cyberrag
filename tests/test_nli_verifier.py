"""Unit tests for CTIEvidenceVerifier."""
import unittest
from src.verification.evidence_verifier import CTIEvidenceVerifier, NLIPrediction


class TestCTIEvidenceVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize verifier once across all tests in this suite
        cls.verifier = CTIEvidenceVerifier()

    def test_01_direct_entailment_support(self):
        evidence = "The Emotet bots reached out to their controllers and received commands to download and execute Trickbot on victim machines."
        claim = "Emotet uses Trickbot."

        pred = self.verifier.classify_pair(evidence=evidence, claim=claim)
        self.assertIsInstance(pred, NLIPrediction)
        self.assertEqual(pred.label, "entailment")
        self.assertGreater(pred.p_entailment, 0.80)
        self.assertLess(pred.p_contradiction, 0.05)

    def test_02_direct_contradiction_conflict(self):
        evidence = "The Emotet bots reached out to their controllers and received commands to download and execute Trickbot on victim machines."
        claim = "Emotet never downloaded or executed Trickbot."

        pred = self.verifier.classify_pair(evidence=evidence, claim=claim)
        self.assertIsInstance(pred, NLIPrediction)
        self.assertEqual(pred.label, "contradiction")
        self.assertGreater(pred.p_contradiction, 0.85)
        self.assertLess(pred.p_entailment, 0.05)

    def test_03_neutral_unrelated_or_unestablished(self):
        evidence = "Emotet was first identified in 2014, while Trickbot emerged separately in 2016 as a banking trojan."
        claim = "Emotet deployed Trickbot to exfiltrate financial credentials."

        pred = self.verifier.classify_pair(evidence=evidence, claim=claim)
        self.assertIsInstance(pred, NLIPrediction)
        self.assertEqual(pred.label, "neutral")
        self.assertGreater(pred.p_neutral, 0.50)

    def test_04_batch_inference_consistency(self):
        pairs = [
            (
                "Emotet was distributing Trickbot with the group tags mor12 and mor13.",
                "Emotet uses Trickbot."
            ),
            (
                "Emotet operations ceased and did not distribute Trickbot in December.",
                "Emotet actively distributed Trickbot in December."
            ),
            (
                "Both Emotet and Trickbot are well known malware families in Eastern Europe.",
                "Emotet exploited CVE-2021-44228 to spread Trickbot."
            )
        ]

        batch_preds = self.verifier.classify_batch(pairs)
        self.assertEqual(len(batch_preds), 3)
        self.assertEqual(batch_preds[0].label, "entailment")
        self.assertEqual(batch_preds[1].label, "contradiction")
        self.assertEqual(batch_preds[2].label, "neutral")


if __name__ == "__main__":
    unittest.main()
