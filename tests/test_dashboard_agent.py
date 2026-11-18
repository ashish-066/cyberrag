"""
Unit and integration tests for CyberRAG Agent Dashboard and Decision Trace.
Validates the data contract between CyberRAGAgent and the Streamlit / Workbench UI.
"""
import unittest
from src.agent.retrieval_agent import CyberRAGAgent, GroundedReport
from src.agent.agent_state import AgentState, AgentStepTrace


class MockRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        if "emotet" in query.lower() and "banking" in query.lower():
            return {
                "dense": [{
                    "report_id": "REP-ZSCALER-01",
                    "segment_id": "p2::s1",
                    "text": "Emotet malware was first observed in 2014 and focused on banking organizations.",
                    "source": "Zscaler",
                    "publication_date": "2021-11-16",
                }],
                "hybrid": [],
                "graph": [],
            }
        return {"dense": [], "hybrid": [], "graph": []}


class TestDashboardAgentIntegration(unittest.TestCase):
    """Verifies that the agent trace and report conform to the UI visualization contract."""

    def setUp(self):
        self.retriever = MockRetriever()
        self.agent = CyberRAGAgent(retriever=self.retriever, max_iterations=3)

    def test_decision_trace_data_contract_verified(self):
        """Ensure all fields required by the Streamlit decision trace UI exist and are populated."""
        report = self.agent.run("Does Emotet target banking organizations?")
        
        self.assertIsInstance(report, GroundedReport)
        self.assertEqual(report.status, "VERIFIED")
        self.assertGreater(report.confidence_score, 0.5)
        self.assertGreaterEqual(report.source_diversity, 1)

        # Check state history
        self.assertIn(AgentState.QUERY_ANALYSIS, report.state_history)
        self.assertIn(AgentState.RETRIEVAL, report.state_history)
        self.assertIn(AgentState.EVIDENCE_INSPECTION, report.state_history)
        self.assertIn(AgentState.VERIFICATION, report.state_history)
        self.assertIn(AgentState.ANSWER_GENERATION, report.state_history)

        # Check execution trace items
        self.assertGreaterEqual(len(report.execution_trace), 4)
        for step in report.execution_trace:
            self.assertIsInstance(step, AgentStepTrace)
            self.assertIsInstance(step.step_index, int)
            self.assertIsInstance(step.state, AgentState)
            self.assertIsInstance(step.thought, str)
            self.assertTrue(len(step.thought) > 0)
            self.assertIsInstance(step.tool_name, str)
            self.assertIsInstance(step.tool_output_summary, str)
            self.assertIsInstance(step.tool_input, dict)

    def test_decision_trace_insufficient_evidence_flow(self):
        """Ensure zero-corpus queries safely step through retries and terminate with audit trail."""
        report = self.agent.run("Does Sandworm use Cyclops Blink?")
        
        self.assertEqual(report.status, "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report.confidence_score, 0.0)
        self.assertEqual(len(report.supporting_evidence), 0)

        # Must record RETRY_EXPANSION steps
        self.assertIn(AgentState.RETRY_EXPANSION, report.state_history)
        self.assertIn(AgentState.INSUFFICIENT_EVIDENCE, report.state_history)
        
        # Verify narrative contains explicit audit findings for Streamlit rendering
        self.assertIn("INSUFFICIENT_EVIDENCE", report.narrative)
        self.assertIn("Entity-Aware Evidence Gate", report.narrative)


if __name__ == "__main__":
    unittest.main()
