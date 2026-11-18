"""Integration and end-to-end tests for VerificationService and whentrue-cti MCP server.

Exercises:
1. Core service construction via bootstrap.build_service (validating StandardRAGRetriever cache arguments)
2. Canonical entity resolution (get_entity)
3. End-to-end claim verification using DeBERTa-v3 NLI (verify_claim)
4. STIX 2.1 bundle generation (export_stix)
5. MCP server tool registry and tool execution guardrails
"""
import os
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.core.config import Settings
from src.core.bootstrap import build_service
from src.core.models import ClaimVerification
from mcp_server.server import ServiceHolder, build_server
from mcp_server.security import Guard
from mcp_server.tools import register_tools
from mcp import types


class TestMCPVerificationService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = Settings()
        cls.service = build_service(cls.settings)

    def test_01_build_service_and_data_integrity(self):
        """Verify build_service successfully loads retriever, verifier, and aliases without keyword argument errors."""
        self.assertIsNotNone(self.service.retriever)
        self.assertIsNotNone(self.service.verifier)
        self.assertIsNotNone(self.service.aliases)
        self.assertTrue(hasattr(self.service.retriever.dense, "embeddings"))
        self.assertEqual(len(self.service.retriever.dense.documents), 5991)

    def test_02_get_entity(self):
        """Verify entity lookup returns canonical info and MITRE-linked category."""
        info = self.service.get_entity("APT28")
        self.assertEqual(info.canonical_name, "APT28")
        self.assertEqual(info.category, "ThreatActor")
        self.assertGreater(info.mention_count, 0)
        self.assertTrue(len(info.neighbors) > 0)

    def test_03_verify_claim_e2e(self):
        """Verify real-model end-to-end claim verification on APT28 USES Zebrocy."""
        result = self.service.verify_claim("APT28", "USES", "Zebrocy")
        self.assertIsInstance(result, ClaimVerification)
        self.assertEqual(result.status, "SUPPORTED")
        self.assertEqual(result.support_tier, "SINGLE_SOURCE")
        self.assertGreater(len(result.supporting), 0)
        self.assertTrue(result.verification_id.startswith("v_"))

    def test_04_export_stix_bundle(self):
        """Verify STIX 2.1 bundle export from verified claim."""
        result = self.service.verify_claim("APT28", "USES", "Zebrocy")
        bundle = self.service.export_stix([result.verification_id])
        self.assertEqual(bundle.get("type"), "bundle")
        objects = bundle.get("objects", [])
        self.assertGreater(len(objects), 0)
        types_present = {obj.get("type") for obj in objects}
        self.assertIn("threat-actor", types_present)
        self.assertIn("malware", types_present)
        self.assertIn("relationship", types_present)

    def test_05_mcp_server_tools_registered(self):
        """Verify all 8 expected tools are registered on the whentrue-cti MCPServer."""
        holder = ServiceHolder(service=self.service)
        server = build_server(holder, self.settings)
        tool_names = set(server._tool_manager._tools.keys())
        expected = {
            "search_evidence",
            "verify_claim",
            "verify_question",
            "get_entity",
            "get_timeline",
            "ingest_report_preview",
            "ingest_report_confirm",
            "export_stix",
        }
        self.assertTrue(expected.issubset(tool_names), f"Missing tools: {expected - tool_names}")

    def test_06_mcp_verify_claim_tool_execution(self):
        """Verify calling the registered MCP verify_claim tool function produces a valid CallToolResult."""
        holder = ServiceHolder(service=self.service)
        server = build_server(holder, self.settings)
        tool_obj = server._tool_manager.get_tool("verify_claim")
        self.assertIsNotNone(tool_obj)
        
        # Execute tool function directly
        result = tool_obj.fn(subject="APT28", relation="USES", object="Zebrocy")
        self.assertIsInstance(result, types.CallToolResult)
        self.assertFalse(getattr(result, "is_error", False))
        self.assertTrue(len(result.content) > 0)
        text_content = result.content[0].text
        self.assertIn("SUPPORTED", text_content)
        self.assertIn("SINGLE_SOURCE", text_content)
        self.assertEqual(result.structured_content["status"], "SUPPORTED")

    def test_07_emotet_trickbot_clean_corpus_by_default(self):
        """Verify that by default (load_demo_overlay=False), MCP service evaluates on genuine corpus text with zero synthetic records."""
        result = self.service.verify_claim("Emotet", "USES", "TrickBot")
        self.assertEqual(result.status, "CONFLICTING_EVIDENCE")
        all_evidence = result.supporting + result.contradicting
        self.assertTrue(all(not c.synthetic for c in all_evidence), "Synthetic records found when overlay is disabled")

    def test_08_emotet_trickbot_with_demo_overlay(self):
        """Verify that when load_demo_overlay=True is set, the synthetic post-takedown record is attached and evaluated."""
        settings = Settings(load_demo_overlay=True)
        demo_svc = build_service(settings)
        result = demo_svc.verify_claim("Emotet", "USES", "TrickBot")
        self.assertEqual(result.status, "CONFLICTING_EVIDENCE")
        self.assertTrue(any(c.synthetic and "TAKEDOWN" in (c.report_id or "") for c in result.contradicting))


if __name__ == "__main__":
    unittest.main()
