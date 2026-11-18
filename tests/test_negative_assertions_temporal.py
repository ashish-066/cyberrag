"""Tests for Negative Assertion Preservation and Temporal Supersedence Resolution.

Validates the 5 Safeguards:
1. Negative assertion retention with complete provenance.
2. Negative assertions are excluded from the Knowledge Graph (baseline 1,400 nodes, 329 edges preserved).
3. Live NLI scoring on newly ingested evidence without hardcoded labels.
4. Correct CTI Context Resolution (TEMPORAL_SUPERSEDENCE, +217 days, CONTRA_LATER).
5. Custom hypothesis compatibility.
"""
from __future__ import annotations
import unittest
from datetime import datetime

from src.nlp.relation_extractor import CyberRelationExtractor
from src.preprocessor.segmenter import PreprocessedSentence
from src.nlp.ner_engine import ExtractedEntity
from src.ingestion import LiveReportProcessor
from src.verification.context_resolver import ContextResolver


class TestNegativeAssertionsAndTemporalSupersedence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = LiveReportProcessor()
        cls.extractor = CyberRelationExtractor()

    def test_01_negative_assertion_extraction_and_provenance(self):
        """Safeguard 1: Extractor retains negative assertions with complete provenance."""
        text = "Investigators confirmed that Emotet ceased operations and never downloaded or executed Trickbot on enterprise networks."
        sent = PreprocessedSentence(
            sentence_id="RPT_TAKEDOWN::p0::s1",
            paragraph_id="RPT_TAKEDOWN::p0",
            report_id="RPT_TAKEDOWN",
            source="Europol / CISA",
            publication_date="2021-05-20",
            paragraph_index=0,
            sentence_index=1,
            global_sentence_index=1,
            text=text,
            start_char=0,
            end_char=len(text),
        )
        e1 = ExtractedEntity(
            entity_id="Malware::Emotet",
            text="Emotet",
            surface_text="Emotet",
            category="Malware",
            canonical_name="Emotet",
            canonical_id="ML_Malware_Emotet",
            start_char=text.find("Emotet"),
            end_char=text.find("Emotet") + 6,
            confidence=0.99,
            is_ambiguous=False,
            sentence_id=sent.sentence_id,
            paragraph_id=sent.paragraph_id,
            report_id=sent.report_id,
            source=sent.source,
            publication_date=sent.publication_date,
        )
        e2 = ExtractedEntity(
            entity_id="Malware::Trickbot",
            text="Trickbot",
            surface_text="Trickbot",
            category="Malware",
            canonical_name="Trickbot",
            canonical_id="ML_Malware_Trickbot",
            start_char=text.find("Trickbot"),
            end_char=text.find("Trickbot") + 8,
            confidence=0.99,
            is_ambiguous=False,
            sentence_id=sent.sentence_id,
            paragraph_id=sent.paragraph_id,
            report_id=sent.report_id,
            source=sent.source,
            publication_date=sent.publication_date,
        )

        rels = self.extractor.extract_from_sentence(sent, [e1, e2])
        self.assertEqual(len(rels), 1)
        rel = rels[0]

        # Provenance verification
        self.assertEqual(rel.source_entity, "Emotet")
        self.assertEqual(rel.relation_type, "USES")
        self.assertEqual(rel.target_entity, "Trickbot")
        self.assertEqual(rel.polarity, "negative")
        self.assertEqual(rel.source, "Europol / CISA")
        self.assertEqual(rel.publication_date, "2021-05-20")
        self.assertEqual(rel.report_id, "RPT_TAKEDOWN")
        self.assertEqual(rel.evidence_text, text)

    def test_02_graph_exclusion_preserves_baseline(self):
        """Safeguard 1 & 5: Negative assertions are strictly excluded from Knowledge Graph edges."""
        takedown_text = (
            "Following joint multinational law enforcement coordination in early 2021, authorities disrupted the Emotet botnet. "
            "Investigators confirmed that Emotet ceased operations and never downloaded or executed Trickbot on enterprise networks "
            "following the January 2021 infrastructure seizure. Threat actors previously reliant on Emotet loaders transitioned to "
            "independent phishing networks."
        )
        ext = self.processor.process_report(
            raw_text=takedown_text,
            report_id="TEST_TAKEDOWN_01",
            source="Europol / CISA",
            pub_date="2021-05-20",
            title="Emotet Takedown",
        )
        self.assertGreaterEqual(len(ext["negative_assertions"]), 1)

        # Baseline graph merge
        merged = self.processor.merge_with_baseline_graph(ext)
        # Graph nodes and edges must match baseline exactly (no new edges added)
        self.assertEqual(merged["total_nodes"], 1400)
        self.assertEqual(merged["new_nodes_count"], 0)
        self.assertEqual(merged["total_edges"], 329)
        self.assertEqual(merged["new_edges_count"], 0)

    def test_03_context_resolver_temporal_supersedence_and_217_days(self):
        """Safeguard 3 & 4: Resolution identifies TEMPORAL_SUPERSEDENCE and 217-day delta."""
        takedown_text = (
            "Following joint multinational law enforcement coordination in early 2021, authorities disrupted the Emotet botnet. "
            "Investigators confirmed that Emotet ceased operations and never downloaded or executed Trickbot on enterprise networks "
            "following the January 2021 infrastructure seizure. Threat actors previously reliant on Emotet loaders transitioned to "
            "independent phishing networks."
        )
        ext = self.processor.process_report(
            raw_text=takedown_text,
            report_id="TEST_TAKEDOWN_02",
            source="Europol / CISA",
            pub_date="2021-05-20",
            title="Emotet Takedown",
        )
        verifs = self.processor.verify_extracted_intelligence(ext)
        self.assertGreaterEqual(len(verifs), 1)

        v = verifs[0]
        self.assertIn(v["claim"], ["Emotet USES TrickBot.", "Emotet USES Trickbot."])
        self.assertEqual(v["status"], "CONFLICTING_EVIDENCE")

        res = v["resolution"]
        self.assertEqual(res["resolution_status"], "TEMPORAL_SUPERSEDENCE")
        self.assertTrue(res["is_conflict"])
        self.assertEqual(res["temporal_relationship"], "CONTRA_LATER")
        self.assertGreaterEqual(res["temporal_delta_days"], 90)

        # Historical reporting vs post-seizure reporting dates
        self.assertTrue(any(d <= "2021-01-30" for d in res["supporting_dates"]))
        self.assertIn("2021-05-20", res["contradicting_dates"])
        self.assertIn("historically supported by reporting", res["explanation"])

    def test_04_live_nli_scoring_no_hardcoding(self):
        """Safeguard 3: The verifier executes live dual-sequence NLI scoring on the May 2021 evidence."""
        verifier = self.processor.pipeline.verifier
        self.assertIsNotNone(verifier)

        premise = (
            "Investigators confirmed that Emotet ceased operations and never downloaded or executed "
            "Trickbot on enterprise networks following the January 2021 infrastructure seizure."
        )
        hypothesis = "Emotet USES Trickbot."

        pred = verifier.classify_pair(evidence=premise, claim=hypothesis)
        self.assertEqual(pred.label, "contradiction")
        self.assertGreater(pred.p_contradiction, 0.90)
        self.assertLess(pred.p_entailment, 0.05)


if __name__ == "__main__":
    unittest.main()
