"""Master Test Runner for CyberRAG / whentrue-cti.

Discovers every tests/**/test_*.py suite (unittest and pytest-style classes both
subclass unittest.TestCase) and runs them:
- tests/test_*.py        legacy v5 suites: preprocessing, NER, relations, KG, retrieval metrics, NLI, resolver
- tests/core/            service, aliases, planner (40-question set), data and hardening regressions
- tests/conditions/      config/data integrity, uploads, sessions, concurrency, fuzz, prompt-injection set
- tests/mcp/             MCP contract and security (S1-S8)
- tests/e2e/             real-model end-to-end checks (skipped if models are not cached or CTI_RUN_REAL=0)

Prefer `python -m pytest`; this runner exists for environments without pytest.
"""

import os
import sys
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py", top_level_dir=".")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
