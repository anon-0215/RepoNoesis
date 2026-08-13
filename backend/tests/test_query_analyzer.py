from __future__ import annotations

import unittest

from app.services.query_analyzer import QueryAnalyzer


class QueryAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = QueryAnalyzer()

    def test_locate_extracts_qualified_symbol_and_auditable_reasons(self):
        result = self.analyzer.analyze("请定位 `Context.invoke` 在哪个文件中定义？")

        self.assertEqual(result.primary_intent, "locate")
        self.assertGreaterEqual(result.confidence, 0.8)
        self.assertEqual(result.symbol_hints, ("Context.invoke",))
        self.assertIn("locate_definition_phrase", result.reason_codes)
        self.assertIn("backticked_identifier", result.reason_codes)
        self.assertIn("qualified_symbol_detected", result.reason_codes)
        self.assertFalse(result.neutral_fallback)

    def test_explain_supports_english_chinese_and_mixed_queries(self):
        queries = (
            "How does `get_current_context` work, and why?",
            "解释 get_current_context 如何工作",
            "请 explain `Context.invoke` 的实现机制",
        )

        for query in queries:
            with self.subTest(query=query):
                result = self.analyzer.analyze(query)
                self.assertEqual(result.primary_intent, "explain")
                self.assertIn("explain_mechanism_phrase", result.reason_codes)

    def test_impact_and_relation_have_directional_soft_hints(self):
        impact = self.analyzer.analyze("修改 get_current_context 后会影响什么，谁会受影响？")
        relation = self.analyzer.analyze("Who calls Context.invoke and what does it call?")

        self.assertEqual(impact.primary_intent, "impact")
        self.assertIn("impact_change_phrase", impact.reason_codes)
        self.assertEqual(impact.routing_hint.relation_direction, "both")
        self.assertEqual(relation.primary_intent, "relation")
        self.assertIn("relation_caller_phrase", relation.reason_codes)
        self.assertEqual(relation.routing_hint.relation_direction, "both")

    def test_multiple_intents_conflict_and_mixed_use_neutral_fallback(self):
        result = self.analyzer.analyze(
            "Where is Context.invoke defined and who calls it, and why does it work?"
        )

        self.assertEqual(result.primary_intent, "mixed")
        self.assertEqual(result.secondary_intents, ("locate", "explain", "relation"))
        self.assertTrue(result.neutral_fallback)
        self.assertIn("multiple_intents_detected", result.reason_codes)
        self.assertIn("conflicting_intent_rules", result.reason_codes)
        self.assertIn("neutral_strategy_fallback", result.reason_codes)
        self.assertEqual(result.routing_hint.dense_weight, 1.0)
        self.assertEqual(result.routing_hint.lexical_weight, 1.0)
        self.assertEqual(result.routing_hint.symbol_weight, 1.0)

    def test_unknown_and_low_confidence_use_neutral_fallback(self):
        result = self.analyzer.analyze("Tell me something interesting about this project.")

        self.assertEqual(result.primary_intent, "unknown")
        self.assertLess(result.confidence, 0.5)
        self.assertTrue(result.neutral_fallback)
        self.assertIn("low_confidence_fallback", result.reason_codes)
        self.assertIn("unknown_intent_fallback", result.reason_codes)

    def test_all_routes_keep_dense_lexical_and_symbol_weights_positive(self):
        queries = (
            "Where is `Context.invoke` defined?",
            "How does `Context.invoke` work?",
            "What changes if `Context.invoke` is modified?",
            "Who calls `Context.invoke`?",
            "Where is `Context.invoke` and who calls it?",
            "Tell me about the project.",
        )

        for query in queries:
            with self.subTest(query=query):
                route = self.analyzer.analyze(query).routing_hint
                self.assertGreater(route.dense_weight, 0.0)
                self.assertGreater(route.lexical_weight, 0.0)
                self.assertGreater(route.symbol_weight, 0.0)

    def test_case_punctuation_and_repeated_input_are_deterministic(self):
        first = self.analyzer.analyze("WHERE is `HTTPServer` DEFINED!!!")
        second = self.analyzer.analyze("where IS `HTTPServer` defined?")

        self.assertEqual(first.primary_intent, "locate")
        self.assertEqual(second.primary_intent, "locate")
        self.assertEqual(first.symbol_hints, second.symbol_hints)
        self.assertEqual(first.reason_codes, second.reason_codes)
        self.assertEqual(first.to_dict(), self.analyzer.analyze(
            "WHERE is `HTTPServer` DEFINED!!!"
        ).to_dict())

    def test_plain_natural_language_is_not_promoted_to_symbol_hints(self):
        result = self.analyzer.analyze(
            "The context is useful and the runner returns a result option."
        )

        self.assertEqual(result.symbol_hints, ())
        self.assertNotIn("camel_case_identifier", result.reason_codes)


if __name__ == "__main__":
    unittest.main()
