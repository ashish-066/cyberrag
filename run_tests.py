"""Run tests using Python built-in unittest."""

import sys
import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Import test functions and adapt to unittest
    from tests.test_preprocessor import (
        test_cleaner_defanging,
        test_cleaner_unicode_normalization,
        test_segmenter_provenance_preservation,
    )
    from tests.test_ner import (
        test_entity_normalizer,
        test_ner_engine_extraction,
    )
    from tests.test_relation_extractor import (
        test_relation_extraction_uses,
        test_relation_extraction_exploits,
        test_relation_extraction_targets,
    )
    from tests.test_knowledge_graph import (
        test_kg_builder_and_analytics,
        sample_graph_data,
    )
    import tempfile
    from pathlib import Path

    class CyberRAGTests(unittest.TestCase):
        def test_1_cleaner_defanging(self):
            test_cleaner_defanging()

        def test_2_cleaner_unicode_normalization(self):
            test_cleaner_unicode_normalization()

        def test_3_segmenter_provenance_preservation(self):
            test_segmenter_provenance_preservation()

        def test_4_entity_normalizer(self):
            test_entity_normalizer()

        def test_5_ner_engine_extraction(self):
            test_ner_engine_extraction()

        def test_6_relation_extraction_uses(self):
            test_relation_extraction_uses()

        def test_7_relation_extraction_exploits(self):
            test_relation_extraction_exploits()

        def test_8_relation_extraction_targets(self):
            test_relation_extraction_targets()

        def test_9_knowledge_graph_builder(self):
            data = sample_graph_data()
            with tempfile.TemporaryDirectory() as tmp_dir:
                test_kg_builder_and_analytics(data, Path(tmp_dir))

        def test_10_benchmark_evaluator(self):
            from src.nlp.evaluator import BenchmarkEvaluator
            evaluator = BenchmarkEvaluator()
            report = evaluator.evaluate()
            ner_perf = report["comparative_ner_performance"]["hybrid_contextual_ml_ner"]["OVERALL"]
            rel_perf = report["relation_extraction_performance"]["OVERALL"]
            self.assertGreaterEqual(ner_perf["f1_score"], 0.85)
            self.assertGreaterEqual(rel_perf["f1_score"], 0.75)
            self.assertEqual(report["evaluation_overview"]["provenance_metadata_failures"], 0)

        def test_11_pipeline_verification(self):
            import verify_pipeline
            success = verify_pipeline.verify()
            self.assertTrue(success)

        def test_12_unseen_cti_evaluation(self):
            from src.nlp.unseen_evaluator import UnseenCTIEvaluator
            evaluator = UnseenCTIEvaluator()
            results = evaluator.evaluate()
            self.assertEqual(results["unseen_evaluation_overview"]["unseen_reports_count"], 4)
            self.assertGreaterEqual(results["novel_entity_discovery_rate"]["total_novel_entities_discovered"], 3)
            self.assertEqual(results["unseen_evaluation_overview"]["provenance_metadata_failures"], 0)

        def test_13_annoctr_evaluation(self):
            from tests.test_annoctr import TestAnnoCTREvaluator
            test_case = TestAnnoCTREvaluator()
            test_case.test_annoctr_evaluation()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CyberRAGTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
