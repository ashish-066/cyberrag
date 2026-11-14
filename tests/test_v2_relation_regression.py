"""Regression tests for v2 CyberRelationExtractor."""
import unittest
from src.nlp.relation_extractor import CyberRelationExtractor
from src.preprocessor.segmenter import PreprocessedSentence
from src.nlp.ner_engine import ExtractedEntity


class TestV2RelationExtractorRegression(unittest.TestCase):
    def setUp(self):
        self.extractor = CyberRelationExtractor()

    def _make_sentence(self, text: str) -> PreprocessedSentence:
        return PreprocessedSentence(
            sentence_id="test::p0::s0",
            paragraph_id="test::p0",
            report_id="test",
            source="Test Vendor",
            publication_date="2021-01-01",
            paragraph_index=0,
            sentence_index=0,
            global_sentence_index=0,
            text=text,
            start_char=0,
            end_char=len(text),
        )

    def _make_entity(self, text: str, category: str, canonical_name: str, start: int, end: int) -> ExtractedEntity:
        return ExtractedEntity(
            entity_id=f"{category}::{canonical_name}",
            text=text,
            surface_text=text,
            category=category,
            canonical_name=canonical_name,
            canonical_id=f"ML_{category}_{canonical_name}",
            start_char=start,
            end_char=end,
            confidence=0.99,
            is_ambiguous=False,
            sentence_id="test::p0::s0",
            paragraph_id="test::p0",
            report_id="test",
            source="Test Vendor",
            publication_date="2021-01-01",
        )

    def test_01_emotet_download_execute_trickbot(self):
        text = "The Emotet bots reached out to their controllers and received commands to download and execute Trickbot on victim machines."
        s = self._make_sentence(text)
        e1 = self._make_entity("Emotet", "Malware", "Emotet", 4, 10)
        e2 = self._make_entity("Trickbot", "Malware", "Trickbot", 95, 103)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, "Emotet")
        self.assertEqual(rels[0].relation_type, "USES")
        self.assertEqual(rels[0].target_entity, "Trickbot")
        self.assertEqual(rels[0].polarity, "positive")

    def test_02_emotet_distributing_trickbot(self):
        text = "Emotet was distributing Trickbot with the group tags mor12 and mor13."
        s = self._make_sentence(text)
        e1 = self._make_entity("Emotet", "Malware", "Emotet", 0, 6)
        e2 = self._make_entity("Trickbot", "Malware", "Trickbot", 25, 33)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, "Emotet")
        self.assertEqual(rels[0].relation_type, "USES")
        self.assertEqual(rels[0].target_entity, "Trickbot")
        self.assertEqual(rels[0].polarity, "positive")

    def test_03_emotet_observed_delivering_trickbot(self):
        text = "In addition, Emotet has been observed delivering Trickbot as a second-stage payload."
        s = self._make_sentence(text)
        e1 = self._make_entity("Emotet", "Malware", "Emotet", 13, 19)
        e2 = self._make_entity("Trickbot", "Malware", "Trickbot", 49, 57)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, "Emotet")
        self.assertEqual(rels[0].relation_type, "USES")
        self.assertEqual(rels[0].target_entity, "Trickbot")
        self.assertEqual(rels[0].polarity, "positive")

    def test_04_clause_boundary_preservation(self):
        text = "While Emotet was active, Trickbot operators were arrested in Europe."
        s = self._make_sentence(text)
        e1 = self._make_entity("Emotet", "Malware", "Emotet", 6, 12)
        e2 = self._make_entity("Trickbot", "Malware", "Trickbot", 25, 33)
        e3 = self._make_entity("Europe", "Location", "Europe", 61, 67)
        rels = self.extractor.extract_from_sentence(s, [e1, e2, e3])
        # Clause boundary 'While ...,' prevents Emotet from binding to Trickbot
        emotet_trickbot_rels = [r for r in rels if r.source_entity == "Emotet" and r.target_entity == "Trickbot"]
        self.assertEqual(len(emotet_trickbot_rels), 0)

    def test_05_negation_suppression(self):
        text = "Emotet did not download Trickbot."
        s = self._make_sentence(text)
        e1 = self._make_entity("Emotet", "Malware", "Emotet", 0, 6)
        e2 = self._make_entity("Trickbot", "Malware", "Trickbot", 25, 33)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        # Sentence contains negation 'did not', must return NO positive assertion
        pos_rels = [r for r in rels if r.polarity == "positive"]
        self.assertEqual(len(pos_rels), 0)

    def test_06_vendor_control_no_targets_on_researcher(self):
        text = "Microsoft researchers observed Emotet downloading files."
        s = self._make_sentence(text)
        e1 = self._make_entity("Microsoft", "Organization", "Microsoft", 0, 9)
        e2 = self._make_entity("Emotet", "Malware", "Emotet", 31, 37)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        # Microsoft is the reporting vendor, should not be claimed as victim or attacker
        ms_rels = [r for r in rels if r.source_entity == "Microsoft" or r.target_entity == "Microsoft"]
        self.assertEqual(len(ms_rels), 0)

    def test_07_vendor_observing_malware_pair(self):
        text = "Microsoft researchers observed Emotet downloading Trickbot."
        s = self._make_sentence(text)
        e1 = self._make_entity("Microsoft", "Organization", "Microsoft", 0, 9)
        e2 = self._make_entity("Emotet", "Malware", "Emotet", 31, 37)
        e3 = self._make_entity("Trickbot", "Malware", "Trickbot", 50, 58)
        rels = self.extractor.extract_from_sentence(s, [e1, e2, e3])
        # Emotet USES Trickbot should be extracted
        emotet_rels = [r for r in rels if r.source_entity == "Emotet" and r.target_entity == "Trickbot"]
        self.assertEqual(len(emotet_rels), 1)
        self.assertEqual(emotet_rels[0].relation_type, "USES")
        # Microsoft must NOT be an actor or target
        ms_rels = [r for r in rels if r.source_entity == "Microsoft" or r.target_entity == "Microsoft"]
        self.assertEqual(len(ms_rels), 0)

    def test_08_passive_observed_downloading(self):
        text = "Emotet was observed downloading Trickbot."
        s = self._make_sentence(text)
        e1 = self._make_entity("Emotet", "Malware", "Emotet", 0, 6)
        e2 = self._make_entity("Trickbot", "Malware", "Trickbot", 32, 40)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, "Emotet")
        self.assertEqual(rels[0].relation_type, "USES")
        self.assertEqual(rels[0].target_entity, "Trickbot")

    def test_09_apt28_deployed_zebrocy_across_europe_no_spatial_targets(self):
        text = "APT28 deployed Zebrocy across Europe."
        s = self._make_sentence(text)
        e1 = self._make_entity("APT28", "ThreatActor", "APT28", 0, 5)
        e2 = self._make_entity("Zebrocy", "Malware", "Zebrocy", 15, 22)
        e3 = self._make_entity("Europe", "Location", "Europe", 30, 36)
        rels = self.extractor.extract_from_sentence(s, [e1, e2, e3])

        # APT28 USES Zebrocy must be extracted
        uses_rels = [r for r in rels if r.source_entity == "APT28" and r.target_entity == "Zebrocy" and r.relation_type == "USES"]
        self.assertEqual(len(uses_rels), 1)

        # Zebrocy TARGETS Europe must NOT be extracted from bare 'across'
        mal_loc_rels = [r for r in rels if r.source_entity == "Zebrocy" and r.target_entity == "Europe"]
        self.assertEqual(len(mal_loc_rels), 0)

        # APT28 TARGETS Europe must NOT be extracted from bare 'across'
        actor_loc_rels = [r for r in rels if r.source_entity == "APT28" and r.target_entity == "Europe"]
        self.assertEqual(len(actor_loc_rels), 0)

    def test_10_apt28_targeted_nato(self):
        text = "APT28 targeted NATO organizations."
        s = self._make_sentence(text)
        e1 = self._make_entity("APT28", "ThreatActor", "APT28", 0, 5)
        e2 = self._make_entity("NATO", "Organization", "NATO", 15, 19)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].source_entity, "APT28")
        self.assertEqual(rels[0].relation_type, "TARGETS")
        self.assertEqual(rels[0].target_entity, "NATO")

    def test_11_apt28_deployed_malware_in_germany_no_automatic_targets(self):
        text = "APT28 deployed Zebrocy in Germany."
        s = self._make_sentence(text)
        e1 = self._make_entity("APT28", "ThreatActor", "APT28", 0, 5)
        e2 = self._make_entity("Zebrocy", "Malware", "Zebrocy", 15, 22)
        e3 = self._make_entity("Germany", "Location", "Germany", 26, 33)
        rels = self.extractor.extract_from_sentence(s, [e1, e2, e3])

        # APT28 USES Zebrocy extracted
        self.assertTrue(any(r.source_entity == "APT28" and r.target_entity == "Zebrocy" and r.relation_type == "USES" for r in rels))
        # No automatic TARGETS Germany from spatial 'in'
        self.assertFalse(any(r.target_entity == "Germany" and r.relation_type == "TARGETS" for r in rels))

    def test_12_zebrocy_operated_across_europe_no_targets(self):
        text = "Zebrocy operated across Europe."
        s = self._make_sentence(text)
        e1 = self._make_entity("Zebrocy", "Malware", "Zebrocy", 0, 7)
        e2 = self._make_entity("Europe", "Location", "Europe", 24, 30)
        rels = self.extractor.extract_from_sentence(s, [e1, e2])
        # 'operated across' is spatial, not intentional targeting
        targets_rels = [r for r in rels if r.relation_type == "TARGETS"]
        self.assertEqual(len(targets_rels), 0)

    def test_13_apt28_attacked_nato_in_europe(self):
        text = "APT28 attacked NATO government networks in Europe."
        s = self._make_sentence(text)
        e1 = self._make_entity("APT28", "ThreatActor", "APT28", 0, 5)
        e2 = self._make_entity("NATO", "Organization", "NATO", 15, 19)
        e3 = self._make_entity("Europe", "Location", "Europe", 43, 49)
        rels = self.extractor.extract_from_sentence(s, [e1, e2, e3])
        # Direct attack on NATO is targeting
        nato_rels = [r for r in rels if r.source_entity == "APT28" and r.target_entity == "NATO" and r.relation_type == "TARGETS"]
        self.assertEqual(len(nato_rels), 1)


if __name__ == "__main__":
    unittest.main()
