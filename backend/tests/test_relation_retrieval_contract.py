from __future__ import annotations

import math
import unittest

from app.services.relation_retrieval import (
    RELATION_EXPANSION_VERSION,
    RELATION_MODE_EXPAND_V1,
    RELATION_MODE_OFF,
    RELATION_SELECTION_VERSION,
    RELATION_TYPE_POLICIES,
    RELATION_WHITELIST_VERSION,
    RelationExpansionLimits,
    canonical_relation_view,
    relation_path_priority,
    validate_relation_mode,
)


class RelationRetrievalContractTests(unittest.TestCase):
    def test_versions_modes_and_real_m3_whitelist_are_frozen(self):
        self.assertEqual(RELATION_EXPANSION_VERSION, "relation_expansion_v1@1")
        self.assertEqual(RELATION_SELECTION_VERSION, "relation_selection_v1@1")
        self.assertEqual(RELATION_WHITELIST_VERSION, "relation_whitelist_v1@1")
        self.assertEqual(
            set(RELATION_TYPE_POLICIES),
            {"imports", "calls", "references", "defines"},
        )
        self.assertEqual(validate_relation_mode(RELATION_MODE_OFF), RELATION_MODE_OFF)
        self.assertEqual(
            validate_relation_mode(RELATION_MODE_EXPAND_V1, retrieval_version="v2"),
            RELATION_MODE_EXPAND_V1,
        )

    def test_invalid_modes_and_v1_relation_are_rejected_strictly(self):
        for invalid in (
            "",
            "   ",
            "EXPAND_V1",
            " expand_v1",
            None,
            1,
            True,
            [],
            {},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_relation_mode(invalid, retrieval_version="v2")
        with self.assertRaises(ValueError):
            validate_relation_mode("expand_v1", retrieval_version="v1")

    def test_direction_views_preserve_the_real_edge_type(self):
        expected = {
            ("calls", "outgoing"): "calls",
            ("calls", "incoming"): "called_by",
            ("imports", "outgoing"): "imports",
            ("imports", "incoming"): "imported_by",
            ("references", "outgoing"): "references",
            ("references", "incoming"): "referenced_by",
            ("defines", "outgoing"): "defines",
            ("defines", "incoming"): "defined_by",
        }
        for pair, view in expected.items():
            with self.subTest(pair=pair):
                self.assertEqual(canonical_relation_view(*pair), view)
        for relation_type, direction in (("owns", "outgoing"), ("calls", "sideways")):
            with self.assertRaises(ValueError):
                canonical_relation_view(relation_type, direction)

    def test_priority_is_finite_versioned_and_separate_from_rrf(self):
        self.assertAlmostEqual(relation_path_priority(1, 1.0, depth=1), 0.5)
        self.assertAlmostEqual(relation_path_priority(3, 0.8, depth=1), 0.2)
        self.assertAlmostEqual(relation_path_priority(1, 1.0, depth=2), 0.25)
        for rank in (0, -1, 1.5, True):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                relation_path_priority(rank, 1.0)
        for weight in (0.0, -1.0, math.nan, math.inf, -math.inf, None):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                relation_path_priority(1, weight)
        with self.assertRaises(ValueError):
            relation_path_priority(1, 1.0, depth=0)

    def test_limits_are_bounded_and_defaults_match_phase4(self):
        limits = RelationExpansionLimits()
        self.assertEqual(limits.max_relation_seeds, 12)
        self.assertEqual(limits.max_edges_per_seed, 8)
        self.assertEqual(limits.max_relation_rows_total, 96)
        self.assertEqual(limits.max_unique_relation_targets, 24)
        self.assertEqual(limits.max_relation_depth, 1)
        self.assertEqual(limits.max_relation_paths_per_target, 8)
        self.assertEqual(limits.max_relation_warnings, 16)
        self.assertEqual(limits.configured_max_relation_slots, 3)
        self.assertAlmostEqual(limits.relation_fraction_cap, 0.30)
        for kwargs in (
            {"max_relation_seeds": 0},
            {"max_edges_per_seed": 0},
            {"max_relation_rows_total": 0},
            {"max_relation_depth": 2},
            {"relation_fraction_cap": math.nan},
            {"relation_fraction_cap": 0.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RelationExpansionLimits(**kwargs)


if __name__ == "__main__":
    unittest.main()
