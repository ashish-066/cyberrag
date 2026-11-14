"""Unit and integration tests for GraphRAG + NLI Verification pipeline."""
import unittest
from typing import Optional, Dict, Any

from src.graph_rag_pipeline import GraphRAGPipeline
from src.verification.evidence_verifier import NLIPrediction


class MockVerifier:
    """Fast, deterministic in-memory mock for CTIEvidenceVerifier."""
    def __init__(self, mode="entailment", should_raise=False):
        self.mode = mode
        self.should_raise = should_raise
        self.call_history = []

    def classify_pair(self, evidence: str, claim: str) -> NLIPrediction:
        if self.should_raise:
            raise RuntimeError("Simulated NLI engine error")
        self.call_history.append((evidence, claim))

        if self.mode == "entailment":
            return NLIPrediction(
                label="entailment",
                p_entailment=0.97,
                p_contradiction=0.01,
                p_neutral=0.02,
                evidence=evidence,
                claim=claim,
            )
        elif self.mode == "contradiction":
            return NLIPrediction(
                label="contradiction",
                p_entailment=0.01,
                p_contradiction=0.96,
                p_neutral=0.03,
                evidence=evidence,
                claim=claim,
            )
        elif self.mode == "conflict":
            if "not" in evidence.lower() or "never" in evidence.lower() or "never" in claim.lower():
                return NLIPrediction(
                    label="contradiction",
                    p_entailment=0.02,
                    p_contradiction=0.95,
                    p_neutral=0.03,
                    evidence=evidence,
                    claim=claim,
                )
            return NLIPrediction(
                label="entailment",
                p_entailment=0.96,
                p_contradiction=0.02,
                p_neutral=0.02,
                evidence=evidence,
                claim=claim,
            )
        else:
            return NLIPrediction(
                label="neutral",
                p_entailment=0.05,
                p_contradiction=0.05,
                p_neutral=0.90,
                evidence=evidence,
                claim=claim,
            )

    def verify_evidence(self, item: Any, override_claim: Optional[str] = None) -> NLIPrediction:
        if self.should_raise:
            raise RuntimeError("Simulated NLI engine error")

        if hasattr(item, "evidence_text"):
            ev_text = item.evidence_text
            cl_text = override_claim or item.claim or ""
            ev_id = item.evidence_id
            ev_type = item.evidence_type
            aid = item.assertion_id
            rid = item.report_id
            sid = item.segment_id
            src = item.source
            pdate = item.publication_date
            conf = item.confidence
            tlevel = item.trust_level
            is_dup = getattr(item, "is_duplicate", False)
            dup_of = getattr(item, "duplicate_of", None)
        else:
            ev_text = item.get("evidence_text") or item.get("evidence") or item.get("text", "")
            cl_text = override_claim or item.get("claim", "")
            ev_id = item.get("evidence_id")
            ev_type = item.get("evidence_type")
            aid = item.get("assertion_id")
            rid = item.get("report_id")
            sid = item.get("segment_id")
            src = item.get("source")
            pdate = item.get("publication_date")
            conf = item.get("confidence")
            tlevel = item.get("trust_level")
            is_dup = item.get("is_duplicate", False)
            dup_of = item.get("duplicate_of", None)

        pred = self.classify_pair(ev_text, cl_text)
        pred.evidence_id = ev_id
        pred.evidence_type = ev_type
        pred.assertion_id = aid
        pred.report_id = rid
        pred.segment_id = sid
        pred.source = src
        pred.publication_date = pdate
        pred.confidence = conf
        pred.trust_level = tlevel
        pred.is_duplicate = is_dup
        pred.duplicate_of = dup_of
        return pred


class MockRetriever:
    """Mock retriever returning controlled graph and text results."""
    def __init__(self, graph_items=None, dense_items=None, bm25_items=None):
        self.graph_items = graph_items if graph_items is not None else [
            {
                "assertion_id": "A000166",
                "claim_text": "Emotet USES Trickbot.",
                "evidence_sentence": "In addition, Emotet has been observed delivering Trickbot as a payload.",
                "source_vendor": "Zscaler",
                "publication_date": "2021-01-29",
                "trust_level": "trusted",
                "confidence": 0.95,
                "subject_id": "Malware::Emotet",
                "object_id": "Malware::Trickbot",
                "predicate": "USES",
                "report_id": "REP-ZSCALER-01",
                "segment_id": "REP-ZSCALER-01::p12::s2",
            }
        ]
        self.dense_items = dense_items if dense_items is not None else [
            {
                "report_id": "REP-INTEL-01",
                "segment_id": "REP-INTEL-01::p3::s1",
                "text": "Intel 471 researchers confirmed Emotet pushed TrickBot malware.",
                "source": "Intel 471",
                "publication_date": "2020-08-15",
                "score": 0.89,
            }
        ]
        self.bm25_items = bm25_items if bm25_items is not None else []

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return {
            "graph": list(self.graph_items),
            "dense": list(self.dense_items),
            "bm25": list(self.bm25_items),
        }


class TestIntegratedVerification(unittest.TestCase):
    def test_01_verification_disabled_returns_not_run(self):
        retriever = MockRetriever()
        verifier = MockVerifier()
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=False)
        self.assertEqual(res["verification"]["status"], "NOT_RUN")
        self.assertIn("answer", res)
        self.assertIn("context", res)
        self.assertEqual(len(verifier.call_history), 0)

    def test_02_graph_assertion_reaches_nli(self):
        retriever = MockRetriever(dense_items=[])
        verifier = MockVerifier(mode="entailment")
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=True)
        self.assertEqual(res["verification"]["status"], "SUPPORTED")
        claims = res["verification"]["claims"]
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["claim"], "Emotet USES Trickbot.")
        self.assertEqual(claims[0]["supporting_count"], 1)

        ev_res = claims[0]["evidence_results"][0]
        self.assertEqual(ev_res["evidence_type"], "graph_assertion")
        self.assertEqual(ev_res["assertion_id"], "A000166")
        self.assertEqual(ev_res["label"], "entailment")

    def test_03_document_passage_reaches_nli(self):
        retriever = MockRetriever()
        verifier = MockVerifier(mode="entailment")
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=True)
        claims = res["verification"]["claims"]
        doc_evs = [
            e for e in claims[0]["evidence_results"]
            if e.get("evidence_type") == "document_passage"
        ]
        self.assertGreaterEqual(len(doc_evs), 1)
        self.assertEqual(doc_evs[0]["source"], "Intel 471")
        self.assertEqual(doc_evs[0]["report_id"], "REP-INTEL-01")

    def test_04_provenance_retained(self):
        retriever = MockRetriever()
        verifier = MockVerifier(mode="entailment")
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=True)
        claims = res["verification"]["claims"]
        graph_ev = [
            e for e in claims[0]["evidence_results"]
            if e.get("evidence_type") == "graph_assertion"
        ][0]

        self.assertEqual(graph_ev["source"], "Zscaler")
        self.assertEqual(graph_ev["publication_date"], "2021-01-29")
        self.assertEqual(graph_ev["assertion_id"], "A000166")
        self.assertEqual(graph_ev["confidence"], 0.95)
        self.assertEqual(graph_ev["trust_level"], "trusted")

    def test_05_multiple_claims_separate_aggregation(self):
        # Two distinct graph assertions with different claims
        graph_items = [
            {
                "assertion_id": "A01",
                "claim_text": "Emotet USES Trickbot.",
                "evidence_sentence": "Emotet drops Trickbot.",
                "source_vendor": "VendorA",
                "publication_date": "2021-01-01",
                "subject_id": "Emotet",
                "object_id": "Trickbot",
                "predicate": "USES",
            },
            {
                "assertion_id": "A02",
                "claim_text": "APT28 TARGETS NATO.",
                "evidence_sentence": "APT28 targeted NATO summits.",
                "source_vendor": "VendorB",
                "publication_date": "2020-05-01",
                "subject_id": "APT28",
                "object_id": "NATO",
                "predicate": "TARGETS",
            },
        ]
        retriever = MockRetriever(graph_items=graph_items, dense_items=[])
        verifier = MockVerifier(mode="entailment")
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("Test multiple claims", verify=True)
        claims = res["verification"]["claims"]
        self.assertEqual(len(claims), 2)
        claim_texts = {c["claim"] for c in claims}
        self.assertEqual(claim_texts, {"Emotet USES Trickbot.", "APT28 TARGETS NATO."})
        self.assertEqual(claims[0]["status"], "SUPPORTED")
        self.assertEqual(claims[1]["status"], "SUPPORTED")

    def test_06_no_graph_claims_insufficient_evidence(self):
        # Only dense text retrieved, no graph assertions
        retriever = MockRetriever(graph_items=[])
        verifier = MockVerifier()
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("Vague question without graph link", verify=True)
        self.assertEqual(res["verification"]["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIn("No candidate claims", res["verification"]["message"])
        self.assertEqual(len(res["verification"]["claims"]), 0)

    def test_07_nli_exception_preserves_retrieval(self):
        retriever = MockRetriever()
        verifier = MockVerifier(should_raise=True)
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=True)
        # Verification failed gracefully
        self.assertEqual(res["verification"]["status"], "ERROR")
        self.assertIn("Simulated NLI engine error", res["verification"]["error"])
        # Original retrieval components remain intact
        self.assertIn("retrieval_results", res)
        self.assertIn("fused_results", res)
        self.assertIn("answer", res)
        self.assertIn("context", res)

    def test_08_conflicting_predictions_conflicting_evidence(self):
        # Provide one supporting sentence and one conflicting sentence
        graph_items = [
            {
                "assertion_id": "A01",
                "claim_text": "Emotet USES Trickbot.",
                "evidence_sentence": "Emotet actively delivered Trickbot.",
                "source_vendor": "Zscaler",
                "publication_date": "2021-01-29",
                "subject_id": "Emotet",
                "object_id": "Trickbot",
                "predicate": "USES",
            }
        ]
        dense_items = [
            {
                "report_id": "REP03",
                "segment_id": "REP03::p1::s1",
                "text": "Emotet never deployed Trickbot during the campaign.",
                "source": "CounterVendor",
                "publication_date": "2021-02-01",
            }
        ]
        retriever = MockRetriever(graph_items=graph_items, dense_items=dense_items)
        verifier = MockVerifier(mode="conflict")
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=True)
        self.assertEqual(res["verification"]["status"], "CONFLICTING_EVIDENCE")
        claims = res["verification"]["claims"]
        self.assertEqual(claims[0]["status"], "CONFLICTING_EVIDENCE")
        self.assertEqual(claims[0]["supporting_count"], 1)
        self.assertEqual(claims[0]["contradicting_count"], 1)

    def test_09_duplicate_evidence_flagged(self):
        # The same segment is present as both a graph assertion and a dense passage
        shared_report_id = "ANNOCTR-ZSCALER-2021-01-29"
        shared_segment_id = "ANNOCTR-ZSCALER-2021-01-29::p12::s2"
        shared_text = "Emotet was distributing Trickbot across enterprise networks."

        graph_items = [
            {
                "assertion_id": "A000166",
                "claim_text": "Emotet USES Trickbot.",
                "evidence_sentence": shared_text,
                "source_vendor": "Zscaler",
                "publication_date": "2021-01-29",
                "report_id": shared_report_id,
                "segment_id": shared_segment_id,
                "subject_id": "Emotet",
                "object_id": "Trickbot",
                "predicate": "USES",
            }
        ]
        dense_items = [
            {
                "report_id": shared_report_id,
                "segment_id": shared_segment_id,
                "text": shared_text,
                "source": "Zscaler",
                "publication_date": "2021-01-29",
            }
        ]
        retriever = MockRetriever(graph_items=graph_items, dense_items=dense_items)
        verifier = MockVerifier(mode="entailment")
        pipeline = GraphRAGPipeline(retriever=retriever, verifier=verifier)

        res = pipeline.ask("How was TrickBot related to Emotet?", verify=True)
        claims = res["verification"]["claims"]
        claim_rec = claims[0]

        # Both records are preserved (total_evidence == 2)
        self.assertEqual(claim_rec["total_evidence"], 2)
        # One is flagged as duplicate
        self.assertEqual(claim_rec["duplicate_count"], 1)
        self.assertEqual(claim_rec["non_duplicate_supporting_count"], 1)

        # Check the items
        items = claim_rec["evidence_results"]
        dups = [item for item in items if item.get("is_duplicate")]
        self.assertEqual(len(dups), 1)
        self.assertEqual(dups[0]["duplicate_of"], "A000166")


if __name__ == "__main__":
    unittest.main()
