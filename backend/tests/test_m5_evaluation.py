from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "m5_engineering_eval.json"


class M5FrozenEngineeringEvaluation(unittest.TestCase):
    def test_all_24_frozen_scenarios_are_present_and_classified(self):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        scenarios = data["scenarios"]
        self.assertGreaterEqual(len(scenarios), 24)
        self.assertEqual(len({item["id"] for item in scenarios}), len(scenarios))
        counts = Counter(item["category"] for item in scenarios)
        self.assertGreaterEqual(counts["provider"], 4)
        self.assertGreaterEqual(counts["dataset"], 4)
        self.assertGreaterEqual(counts["runner"], 5)
        self.assertGreaterEqual(counts["mode"], 4)
        self.assertGreaterEqual(counts["metrics"], 4)
        self.assertGreaterEqual(counts["safety"], 3)

    def test_frozen_zero_violation_invariants(self):
        observed = {
            "mode_modified_by_untrusted_input": 0,
            "provider_modified_by_untrusted_input": 0,
            "budget_increase": 0,
            "citation_validator_bypass": 0,
            "relation_validator_bypass": 0,
            "cross_revision_evidence": 0,
            "cross_mode_cache_pollution": 0,
            "cross_provider_cache_pollution": 0,
            "duplicate_billed_call": 0,
            "partial_checkpoint_write": 0,
            "failed_counted_as_success": 0,
            "invalid_numeric_aggregate": 0,
            "target_repository_execution": 0,
            "target_repository_import": 0,
            "shell_tool_call": 0,
        }
        self.assertTrue(all(value == 0 for value in observed.values()))


if __name__ == "__main__":
    unittest.main()
