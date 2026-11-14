"""Master Test Runner for CyberRAG.

Discovers and executes all 59 unit tests across all test suites:
- test_preprocessor.py: Indicator refanging, unicode normalization, segmentation provenance
- test_ner.py: Entity normalization and gazetteer extraction
- test_relation_extractor.py: Typed relationship extraction
- test_knowledge_graph.py: NetworkX graph construction, exports, and analytics
- test_annoctr.py: Real-world Bosch Research AnnoCTR benchmark evaluation
- test_unseen_cti.py: Zero-shot generalization and novel entity discovery on 4 fresh reports
- test_retrieval_helpers.py: Retrieval evaluation metrics (Recall@K, MRR)
- test_claim_aggregation.py: Evidence aggregation and multi-publisher consensus
- test_provenance.py: Sentence- and paragraph-level metadata persistence
- test_nli_verifier.py: DeBERTa cross-encoder NLI verification with raw softmax scores
- test_context_resolver.py: 4-tier contradiction taxonomy classification
- test_integrated_verification.py: End-to-end fused GraphRAG verification
- test_negative_assertions_temporal.py: Negative assertion provenance and temporal supersedence
- test_v2_relation_regression.py: Regression testing for vendor and spatial trigger suppression
"""

import sys
import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
