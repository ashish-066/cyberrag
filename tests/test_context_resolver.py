"""Unit tests for ContextResolver module."""
import unittest
from src.verification.context_resolver import ContextResolver, ContradictionResolution


class TestContextResolver(unittest.TestCase):
    def test_01_no_conflict_when_only_supporting(self):
        evidence = [
            {
                "label": "entailment",
                "source": "Zscaler",
                "publication_date": "2021-01-29",
                "evidence": "Emotet distributed Trickbot.",
            },
            {
                "label": "neutral",
                "source": "Intel 471",
                "publication_date": "2020-10-15",
                "evidence": "General botnet background.",
            },
        ]
        res = ContextResolver.resolve("Emotet USES Trickbot.", evidence)
        self.assertIsInstance(res, ContradictionResolution)
        self.assertEqual(res.resolution_status, "NO_CONFLICT")
        self.assertFalse(res.is_conflict)
        self.assertEqual(res.supporting_sources, ["Zscaler"])
        self.assertEqual(res.contradicting_sources, [])

    def test_02_temporal_supersedence_contra_later(self):
        evidence = [
            {
                "label": "entailment",
                "source": "Intel 471",
                "publication_date": "2020-10-15",
                "evidence": "Emotet bots distributed Trickbot on endpoints.",
            },
            {
                "label": "contradiction",
                "source": "Zscaler",
                "publication_date": "2021-06-24",
                "evidence": "Emotet infrastructure was disrupted; no Trickbot payloads were observed.",
            },
        ]
        res = ContextResolver.resolve("Emotet USES Trickbot.", evidence)
        self.assertEqual(res.resolution_status, "TEMPORAL_SUPERSEDENCE")
        self.assertTrue(res.is_conflict)
        self.assertEqual(res.temporal_relationship, "CONTRA_LATER")
        self.assertGreaterEqual(res.temporal_delta_days, 200)
        self.assertIn("evolution", res.explanation.lower())

    def test_03_temporal_supersedence_supp_later(self):
        evidence = [
            {
                "label": "contradiction",
                "source": "VendorA",
                "publication_date": "2019-01-15",
                "evidence": "Actor did not utilize Trickbot payloads in initial campaigns.",
            },
            {
                "label": "entailment",
                "source": "VendorB",
                "publication_date": "2020-11-10",
                "evidence": "Actor adopted Trickbot as primary loader in late 2020.",
            },
        ]
        res = ContextResolver.resolve("Actor USES Trickbot.", evidence)
        self.assertEqual(res.resolution_status, "TEMPORAL_SUPERSEDENCE")
        self.assertTrue(res.is_conflict)
        self.assertEqual(res.temporal_relationship, "SUPP_LATER")
        self.assertGreaterEqual(res.temporal_delta_days, 300)
        self.assertIn("resumed campaign", res.explanation.lower())

    def test_04_genuine_attribution_dispute(self):
        evidence = [
            {
                "label": "entailment",
                "source": "Mandiant",
                "publication_date": "2021-04-10",
                "evidence": "Mandiant attributes the intrustion to APT28.",
            },
            {
                "label": "contradiction",
                "source": "CrowdStrike",
                "publication_date": "2021-04-15",
                "evidence": "CrowdStrike assesses the campaign was conducted by Sandworm, not APT28.",
            },
        ]
        res = ContextResolver.resolve("APT28 ATTRIBUTED_TO Intrusion.", evidence)
        self.assertEqual(res.resolution_status, "GENUINE_ATTRIBUTION_DISPUTE")
        self.assertTrue(res.is_conflict)
        self.assertEqual(res.temporal_relationship, "CONCURRENT")
        self.assertLessEqual(res.temporal_delta_days, 10)
        self.assertIn("Mandiant", res.supporting_sources)
        self.assertIn("CrowdStrike", res.contradicting_sources)

    def test_05_campaign_divergence(self):
        evidence = [
            {
                "label": "entailment",
                "source": "VendorA",
                "publication_date": "2021-01-10",
                "evidence": "In the mor12 campaign, Emotet delivered Trickbot to banking sector victims.",
            },
            {
                "label": "contradiction",
                "source": "VendorA",
                "publication_date": "2021-01-12",
                "evidence": "In the epoch cluster campaign, Emotet did not deliver Trickbot.",
            },
        ]
        res = ContextResolver.resolve("Emotet USES Trickbot.", evidence)
        self.assertEqual(res.resolution_status, "CAMPAIGN_DIVERGENCE")
        self.assertTrue(res.is_conflict)
        self.assertIn("campaign", res.explanation.lower())

    def test_06_ambiguous_conflict_missing_dates(self):
        evidence = [
            {
                "label": "entailment",
                "source": "VendorA",
                "publication_date": None,
                "evidence": "Emotet delivered Trickbot.",
            },
            {
                "label": "contradiction",
                "source": "VendorB",
                "publication_date": "2021-01-20",
                "evidence": "Emotet never delivered Trickbot.",
            },
        ]
        res = ContextResolver.resolve("Emotet USES Trickbot.", evidence)
        self.assertEqual(res.resolution_status, "AMBIGUOUS_CONFLICT")
        self.assertTrue(res.is_conflict)
        self.assertEqual(res.temporal_relationship, "UNKNOWN")
        self.assertFalse(res.resolvable)

    def test_07_empty_claim_raises_error(self):
        with self.assertRaises(ValueError):
            ContextResolver.resolve("", [])

    def test_08_to_dict_serialization(self):
        evidence = [
            {
                "label": "entailment",
                "source": "Zscaler",
                "publication_date": "2021-01-29",
                "evidence": "Evidence 1",
            }
        ]
        res = ContextResolver.resolve("Claim A", evidence)
        d = res.to_dict()
        self.assertEqual(d["claim"], "Claim A")
        self.assertEqual(d["resolution_status"], "NO_CONFLICT")
        self.assertIn("recommended_action", d)


if __name__ == "__main__":
    unittest.main()
