"""Unit test for Unseen Real CTI Reports evaluation."""

import unittest
from pathlib import Path
from src.nlp.unseen_evaluator import UnseenCTIEvaluator


class TestUnseenCTIEvaluator(unittest.TestCase):
    """Test suite for evaluation on fresh, unseen CTI reports."""

    def test_unseen_cti_evaluation(self):
        evaluator = UnseenCTIEvaluator()
        results = evaluator.evaluate()

        overview = results["unseen_evaluation_overview"]
        self.assertEqual(overview["unseen_reports_count"], 4)
        self.assertGreater(overview["total_entities_extracted"], 25)
        self.assertGreater(overview["total_relations_extracted"], 5)
        self.assertEqual(overview["provenance_metadata_failures"], 0)
        self.assertEqual(overview["provenance_compliance_rate"], "100.0%")

        # Test novel entity discovery
        novel_rate = results["novel_entity_discovery_rate"]
        self.assertGreaterEqual(novel_rate["total_novel_entities_discovered"], 3)


if __name__ == "__main__":
    unittest.main()
