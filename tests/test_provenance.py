"""Unit tests for ProvenanceAdapter."""
import unittest
from src.verification.provenance_adapter import ProvenanceAdapter, VerificationEvidence


class TestProvenanceAdapter(unittest.TestCase):
    def test_01_graph_assertion_conversion(self):
        graph_item = {
            "type": "graph",
            "assertion_id": "A000166",
            "claim": "Emotet USES Trickbot.",
            "evidence": "In addition, Emotet has been observed delivering Trickbot as a second-stage payload.",
            "source": "Zscaler",
            "publication_date": "2021-01-29",
            "trust_level": "trusted",
            "confidence": 0.95,
            "verification_status": "unverified",
            "human_verified": False,
            "subject_id": "Malware::Emotet",
            "object_id": "Malware::Trickbot",
            "predicate": "USES"
        }

        evidence = ProvenanceAdapter.adapt_item(graph_item)
        self.assertIsInstance(evidence, VerificationEvidence)
        self.assertEqual(evidence.evidence_id, "A000166")
        self.assertEqual(evidence.evidence_type, "graph_assertion")
        self.assertEqual(evidence.claim, "Emotet USES Trickbot.")
        self.assertEqual(
            evidence.evidence_text,
            "In addition, Emotet has been observed delivering Trickbot as a second-stage payload."
        )
        self.assertEqual(evidence.assertion_id, "A000166")
        self.assertEqual(evidence.source, "Zscaler")
        self.assertEqual(evidence.publication_date, "2021-01-29")
        self.assertEqual(evidence.trust_level, "trusted")
        self.assertEqual(evidence.confidence, 0.95)
        self.assertEqual(evidence.predicate, "USES")

    def test_02_document_passage_conversion(self):
        doc_item = {
            "type": "text",
            "report_id": "ANNOCTR-ZSCALER-2021-01-29-DID-COVID-CANCEL-CHRISTMA",
            "segment_id": "ANNOCTR-ZSCALER-2021-01-29-DID-COVID-CANCEL-CHRISTMA::p12::s2",
            "text": "Emotet was distributing Trickbot with the group tags * mor12 * and * mor13 *.",
            "source": "Zscaler",
            "publication_date": "2021-01-29",
            "retrieval_methods": ["dense", "hybrid"],
            "scores": {"rrf_score": 0.032}
        }

        evidence = ProvenanceAdapter.adapt_item(doc_item, default_claim="Emotet uses Trickbot.")
        self.assertIsInstance(evidence, VerificationEvidence)
        self.assertEqual(
            evidence.evidence_id,
            "ANNOCTR-ZSCALER-2021-01-29-DID-COVID-CANCEL-CHRISTMA::ANNOCTR-ZSCALER-2021-01-29-DID-COVID-CANCEL-CHRISTMA::p12::s2"
        )
        self.assertEqual(evidence.evidence_type, "document_passage")
        self.assertEqual(evidence.evidence_text, "Emotet was distributing Trickbot with the group tags * mor12 * and * mor13 *.")
        self.assertEqual(evidence.claim, "Emotet uses Trickbot.")
        self.assertIsNone(evidence.assertion_id)
        self.assertEqual(evidence.report_id, "ANNOCTR-ZSCALER-2021-01-29-DID-COVID-CANCEL-CHRISTMA")
        self.assertEqual(evidence.segment_id, "ANNOCTR-ZSCALER-2021-01-29-DID-COVID-CANCEL-CHRISTMA::p12::s2")
        self.assertEqual(evidence.source, "Zscaler")
        self.assertEqual(evidence.publication_date, "2021-01-29")
        self.assertIsNone(evidence.confidence)
        self.assertIsNone(evidence.trust_level)

    def test_03_source_vendor_mapping(self):
        item = {
            "type": "text",
            "source_vendor": "Intel 471",
            "text": "Intel 471 researchers tracked the Emotet campaign.",
            "report_id": "REP01",
            "segment_id": "SEG01"
        }
        evidence = ProvenanceAdapter.adapt_item(item)
        self.assertEqual(evidence.source, "Intel 471")

    def test_04_sentence_id_mapping(self):
        item = {
            "type": "text",
            "sentence_id": "REP01::p0::s1",
            "report_id": "REP01",
            "text": "Trickbot was deployed across victim endpoints."
        }
        evidence = ProvenanceAdapter.adapt_item(item)
        self.assertEqual(evidence.segment_id, "REP01::p0::s1")

    def test_05_missing_optional_provenance_remains_none(self):
        item = {
            "type": "graph",
            "evidence": "Trickbot attacked financial institutions.",
            "assertion_id": "A100"
        }
        evidence = ProvenanceAdapter.adapt_item(item)
        self.assertIsNone(evidence.report_id)
        self.assertIsNone(evidence.segment_id)
        self.assertIsNone(evidence.source)
        self.assertIsNone(evidence.publication_date)
        self.assertIsNone(evidence.confidence)
        self.assertIsNone(evidence.trust_level)
        self.assertIsNone(evidence.subject)
        self.assertIsNone(evidence.predicate)
        self.assertIsNone(evidence.object)

    def test_06_missing_evidence_text_raises_value_error(self):
        item = {
            "type": "graph",
            "assertion_id": "A101"
            # Missing evidence text
        }
        with self.assertRaises(ValueError):
            ProvenanceAdapter.adapt_item(item)

        item2 = {
            "type": "text",
            "report_id": "REP02"
            # Missing text
        }
        with self.assertRaises(ValueError):
            ProvenanceAdapter.adapt_item(item2)

    def test_07_unsupported_evidence_type_raises_value_error(self):
        item = {
            "type": "unknown_format",
            "text": "Some text"
        }
        with self.assertRaises(ValueError):
            ProvenanceAdapter.adapt_item(item)

    def test_08_evidence_text_preserved_verbatim(self):
        raw_text = "  Exact whitespace and markdown symbols * * [Link] (https:// example.com)  "
        item = {
            "type": "text",
            "text": raw_text,
            "report_id": "R1",
            "segment_id": "S1"
        }
        evidence = ProvenanceAdapter.adapt_item(item)
        self.assertEqual(evidence.evidence_text, raw_text)


if __name__ == "__main__":
    unittest.main()
