"""Unit test for AnnoCTR Real CTI NER Evaluation."""

import unittest
from pathlib import Path
from src.nlp.annoctr_evaluator import AnnoCTREvaluator


class TestAnnoCTREvaluator(unittest.TestCase):
    """Test suite for independent AnnoCTR real-world CTI benchmark evaluation."""

    def test_annoctr_evaluation(self):
        evaluator = AnnoCTREvaluator()
        report = evaluator.evaluate()

        overview = report["annoctr_evaluation_overview"]
        self.assertGreaterEqual(overview["total_sentences_evaluated"], 100)
        self.assertGreaterEqual(overview["total_gold_entities"], 150)

        # Baseline Gazetteer metrics check (real-world empirical performance)
        baseline_perf = report["baseline_gazetteer_performance"]["OVERALL"]
        self.assertGreaterEqual(baseline_perf["precision"], 0.30)
        self.assertGreaterEqual(baseline_perf["recall"], 0.25)
        self.assertGreaterEqual(baseline_perf["f1_score"], 0.30)

        # Hybrid Contextual ML NER metrics check (demonstrating ML generalization gain)
        hybrid_perf = report["hybrid_contextual_ner_performance"]["OVERALL"]
        self.assertGreaterEqual(hybrid_perf["precision"], 0.35)
        self.assertGreaterEqual(hybrid_perf["recall"], 0.35)
        self.assertGreaterEqual(hybrid_perf["f1_score"], 0.40)

        # Ensure error analysis is populated with genuine out-of-vocabulary misses
        err_analysis = report["error_analysis"]
        self.assertGreater(len(err_analysis["out_of_vocabulary_misses"]), 5)


if __name__ == "__main__":
    unittest.main()
