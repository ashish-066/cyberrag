"""
Unit and integration tests for CyberRAGAgent and its state machine.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import unittest
from typing import Any, Dict, List, Optional
from src.verification.evidence_verifier import CTIEvidenceVerifier
from src.agent.retrieval_agent import (
    CyberRAGAgent,
    AgentState,
    QueryIntent,
    LLMQueryPlanner,
    EntityAwareEvidenceGate,
    SubClaim,
    CANONICAL_PREDICATES,
)


class MockRetriever:
    def __init__(self, mode="normal"):
        self.mode = mode

    def retrieve(self, query: str, top_k: int = 5):
        if self.mode == "empty":
            return {"dense": [], "hybrid": [], "graph": []}
        elif self.mode == "irrelevant":
            # Returns cyber passages mentioning Taurus and Cobalt Strike, but neither Sandworm nor Cyclops Blink
            return {
                "dense": [
                    {
                        "report_id": "REP-01",
                        "segment_id": "p1::s1",
                        "text": "Taurus stealer performs sandbox evasion in victim environments.",
                        "source": "Zscaler",
                        "publication_date": "2021-05-10",
                    }
                ],
                "hybrid": [
                    {
                        "report_id": "REP-02",
                        "segment_id": "p2::s3",
                        "text": "Cobalt Strike beacon injection observed across multiple campaigns.",
                        "source": "Proofpoint",
                        "publication_date": "2020-08-12",
                    }
                ],
                "graph": [],
            }
        elif self.mode == "supported":
            return {
                "dense": [
                    {
                        "report_id": "REP-ZSCALER-01",
                        "segment_id": "p2::s1",
                        "text": "Emotet malware was first observed in 2014 and focused on banking fraud.",
                        "source": "Zscaler",
                        "publication_date": "2021-11-16",
                    }
                ],
                "hybrid": [
                    {
                        "report_id": "REP-PROOFPOINT-02",
                        "segment_id": "p1::s4",
                        "text": "The banking sector was actively targeted by Emotet campaigns throughout late 2021.",
                        "source": "Proofpoint",
                        "publication_date": "2021-11-20",
                    }
                ],
                "graph": [],
            }
        elif self.mode == "temporal_conflict":
            return {
                "dense": [
                    {
                        "report_id": "REP-OCT-2020",
                        "segment_id": "p1::s1",
                        "text": "Emotet deploys TrickBot as a secondary payload for network reconnaissance.",
                        "source": "Intel 471",
                        "publication_date": "2020-10-15",
                    }
                ],
                "hybrid": [
                    {
                        "report_id": "REP-MAY-2021",
                        "segment_id": "p3::s2",
                        "text": "Following global law enforcement disruption, Emotet no longer uses TrickBot payloads.",
                        "source": "Proofpoint",
                        "publication_date": "2021-05-20",
                    }
                ],
                "graph": [],
            }
        return {"dense": [], "hybrid": [], "graph": []}


class StubOfflineLLMClient:
    """Stub client ensuring unit tests are completely deterministic and offline."""
    def is_available(self) -> bool:
        return False

    def chat_complete(self, system: str, prompt: str, **kwargs) -> Optional[str]:
        return None


class MockActiveLLMClient:
    """Mock client simulating live LLM responses to test query planner guardrails."""
    def __init__(self, canned_response: str):
        self.canned_response = canned_response

    def is_available(self) -> bool:
        return True

    def chat_complete(self, system: str, prompt: str, **kwargs) -> Optional[str]:
        return self.canned_response


class TestAgentComponents(unittest.TestCase):
    def setUp(self):
        self.planner = LLMQueryPlanner(llm_client=StubOfflineLLMClient())
        self.gate = EntityAwareEvidenceGate()

    def test_query_planner_intent_and_entities(self):
        plan = self.planner.plan("Does Sandworm use Cyclops Blink?")
        self.assertEqual(plan.intent, QueryIntent.FACTUAL_VERIFICATION)
        self.assertIn("Sandworm", plan.entities)
        self.assertIn("Cyclops", plan.entities[1])
        self.assertEqual(len(plan.sub_claims), 1)
        self.assertEqual(plan.sub_claims[0].predicate, "USES")

    def test_query_planner_canonical_predicate_mapping(self):
        # "targets" -> TARGETS
        plan_t = self.planner.plan("Emotet targets banking organizations")
        self.assertEqual(plan_t.sub_claims[0].predicate, "TARGETS")

        # "exploits" -> EXPLOITS
        plan_e = self.planner.plan("APT28 exploits CVE-2020-0601")
        self.assertEqual(plan_e.sub_claims[0].predicate, "EXPLOITS")

        # "associated with" -> USES
        plan_a = self.planner.plan("Sandworm associated with BlackEnergy")
        self.assertEqual(plan_a.sub_claims[0].predicate, "USES")

    def test_query_planner_guardrails_with_live_llm_simulation(self):
        # Even if a live LLM attempts to return RELATIONAL_QUERY and TARGETS for 'associated with',
        # planner guardrails must strictly enforce canonical CTI semantics.
        mock_resp = json.dumps({
            "intent": "RELATIONAL_QUERY",
            "subject": "Sandworm",
            "predicate": "TARGETS",
            "object": "BlackEnergy"
        })
        llm_planner = LLMQueryPlanner(llm_client=MockActiveLLMClient(mock_resp))
        plan = llm_planner.plan("Sandworm associated with BlackEnergy")
        self.assertEqual(plan.sub_claims[0].predicate, "USES")

        mock_resp2 = json.dumps({
            "intent": "RELATIONAL_QUERY",
            "subject": "Sandworm",
            "predicate": "USES",
            "object": "Cyclops Blink"
        })
        llm_planner2 = LLMQueryPlanner(llm_client=MockActiveLLMClient(mock_resp2))
        plan2 = llm_planner2.plan("Does Sandworm use Cyclops Blink?")
        self.assertEqual(plan2.intent, QueryIntent.FACTUAL_VERIFICATION)
        self.assertEqual(plan2.sub_claims[0].predicate, "USES")

    def test_query_planner_temporal_scope(self):
        plan = self.planner.plan("How did TrickBot evolve between 2019 and 2021?")
        self.assertEqual(plan.intent, QueryIntent.TEMPORAL_EVOLUTION)
        self.assertIsNotNone(plan.temporal_scope)
        self.assertEqual(plan.temporal_scope[0], "2019-01-01")
        self.assertEqual(plan.temporal_scope[1], "2021-12-31")

    def test_entity_aware_evidence_gate(self):
        # 1. Both present -> Probative
        ok, reason = self.gate.is_probative(
            "Sandworm actors deployed Cyclops Blink firmware malware.",
            "Sandworm", "Cyclops Blink"
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "PROBATIVE_COOCCURRENCE")

        # 2. Alias present -> Probative (TeleBots is alias for Sandworm)
        ok_al, _ = self.gate.is_probative(
            "TeleBots operators were linked to Cyclops Blink modules.",
            "Sandworm", "Cyclops Blink"
        )
        self.assertTrue(ok_al)

        # 3. Only subject present -> Non-probative
        ok_s, reason_s = self.gate.is_probative(
            "Sandworm launched NotPetya against regional power grids.",
            "Sandworm", "Cyclops Blink"
        )
        self.assertFalse(ok_s)
        self.assertTrue("MISSING_OBJECT" in reason_s)

        # 4. Irrelevant passage -> Non-probative
        ok_irr, reason_irr = self.gate.is_probative(
            "Taurus stealer evades sandbox detection using timing loops.",
            "Sandworm", "Cyclops Blink"
        )
        self.assertFalse(ok_irr)
        self.assertEqual(reason_irr, "NEITHER_ENTITY_PRESENT")

        # 5. Footnote / Bibliography metadata rejection
        bib_text = "[[1]] (# _ ftnref1) ESET [Analysis of Zebrocy] [[2]] (# _ ftnref2) Kaspersky [Turla Fresh Faces]"
        ok_bib, reason_bib = self.gate.is_probative(bib_text, "Turla", "Zebrocy")
        self.assertFalse(ok_bib)
        self.assertEqual(reason_bib, "BIBLIOGRAPHY_OR_FOOTNOTE_REJECTED")

        # 6. Inline citation marker in clean narrative sentence must PASS
        clean_citation = "Zebrocy is a malware used by APT28 (also known as Sofacy) [[1]] (# _ ftn1) in the last two years."
        ok_clean, reason_clean = self.gate.is_probative(clean_citation, "APT28", "Zebrocy")
        self.assertTrue(ok_clean)
        self.assertEqual(reason_clean, "PROBATIVE_COOCCURRENCE")

        # 7. Partial name overlap rejection (Fancy Lazarus != Lazarus)
        fancy_text = "The extortion threat actor known as Fancy Lazarus is targeting financial sectors."
        ok_fancy, reason_fancy = self.gate.is_probative(fancy_text, "Lazarus", "financial")
        self.assertFalse(ok_fancy)
        self.assertTrue("MISSING_SUBJECT" in reason_fancy)

        # 8. Cross-sentence co-occurrence rejection
        cross_text = "Kaspersky documented Turla operations. Separately, Mandiant tracked Zebrocy samples."
        ok_cross, reason_cross = self.gate.is_probative(cross_text, "Turla", "Zebrocy")
        self.assertFalse(ok_cross)
        self.assertEqual(reason_cross, "CROSS_SENTENCE_COOCCURRENCE_REJECTED")

    def test_hedged_evidence_detection_and_capping(self):
        verifier = CTIEvidenceVerifier()
        # Modal hedging phrase: 'might link' -> hedged, capped
        pred_hedged = verifier.classify_pair(
            "Further, we uncovered new evidence that might link the attacks to Lazarus.",
            "Lazarus targets banks."
        )
        self.assertTrue(pred_hedged.is_hedged)
        self.assertLessEqual(pred_hedged.p_entailment, 0.65)

        # Month 'May' or bare 'could' / 'potential' -> NOT hedged
        pred_may = verifier.classify_pair(
            "On May 20, 2021 Europol announced the disruption of Emotet infrastructure.",
            "Europol disrupted Emotet."
        )
        self.assertFalse(pred_may.is_hedged)

        pred_could = verifier.classify_pair(
            "Attackers could use the harvested credentials to access victim systems.",
            "Attackers access systems."
        )
        self.assertFalse(pred_could.is_hedged)

        # Batch verification test: Ensure classify_batch uses identical narrowed pattern
        batch_preds = verifier.classify_batch([
            (
                "On May 20, 2021 Europol announced the disruption of Emotet infrastructure.",
                "Europol disrupted Emotet.",
            ),
            (
                "Further, we uncovered new evidence that might link the attacks to Lazarus.",
                "Lazarus targets banks.",
            ),
        ])
        self.assertFalse(batch_preds[0].is_hedged)
        self.assertTrue(batch_preds[1].is_hedged)
        self.assertLessEqual(batch_preds[1].p_entailment, 0.65)


class TestCyberRAGAgentExecution(unittest.TestCase):
    def test_agent_insufficient_evidence_exit(self):
        # Scenario: Sandworm USES Cyclops Blink where corpus contains no co-occurring evidence
        retriever = MockRetriever(mode="irrelevant")
        agent = CyberRAGAgent(retriever=retriever, max_iterations=3)

        report = agent.run("Does Sandworm use Cyclops Blink?")

        # Invariant checks:
        self.assertEqual(report.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report.confidence_score, 0.0)
        self.assertEqual(len(report.supporting_evidence), 0)
        self.assertIn(AgentState.QUERY_ANALYSIS, report.state_history)
        self.assertIn(AgentState.RETRIEVAL, report.state_history)
        self.assertIn(AgentState.EVIDENCE_INSPECTION, report.state_history)
        self.assertIn(AgentState.INSUFFICIENT_EVIDENCE, report.state_history)
        # Verify NLI verification was bypassed
        self.assertNotIn(AgentState.VERIFICATION, report.state_history)

    def test_agent_verified_support_with_diversity(self):
        # Scenario: Emotet TARGETS banking with independent vendor sources
        retriever = MockRetriever(mode="supported")
        agent = CyberRAGAgent(retriever=retriever, max_iterations=3)

        report = agent.run("Does Emotet target banking?")

        self.assertEqual(report.status, "VERIFIED")
        self.assertGreater(report.confidence_score, 0.80)
        self.assertEqual(report.source_diversity, 2)  # Zscaler + Proofpoint
        self.assertEqual(len(report.supporting_evidence), 2)
        self.assertIn(AgentState.VERIFICATION, report.state_history)
        self.assertIn(AgentState.ANSWER_GENERATION, report.state_history)

    def test_agent_temporal_supersedence_resolution(self):
        # Scenario: Oct 2020 (deploy) vs May 2021 (no longer deploy) -> 217-day delta
        retriever = MockRetriever(mode="temporal_conflict")
        agent = CyberRAGAgent(retriever=retriever, max_iterations=3, temporal_delta_days=90)

        report = agent.run("Does Emotet deploy TrickBot?")

        self.assertEqual(report.status, "TEMPORAL_SUPERSEDENCE")
        self.assertIsNotNone(report.conflict_resolution)
        self.assertGreater(report.conflict_resolution["delta_days"], 90)
        self.assertIn(AgentState.CONFLICT_RESOLUTION, report.state_history)


if __name__ == "__main__":
    unittest.main()
