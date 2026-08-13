from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from app.database import Database
from app.services.symbol_retriever import SymbolRetriever
from tests.m1_helpers import make_project


class SymbolRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.directory.name) / "symbols.sqlite")
        self.project_id, _bundle = make_project(
            self.database,
            [
                (
                    "src/context.py",
                    "Context.invoke",
                    "def invoke(self, callback):\n    return callback()\n",
                ),
                (
                    "src/other.py",
                    "Other.invoke",
                    "def invoke(self, callback):\n    return callback()\n",
                ),
                (
                    "testing/runner.py",
                    "CliRunner.isolation.should_strip_ansi",
                    "def should_strip_ansi(stream):\n    return stream is None\n",
                ),
                (
                    "src/globals.py",
                    "get_current_context",
                    "def get_current_context():\n    return current_context\n",
                ),
                (
                    "src/server.py",
                    "HTTPServer",
                    "class HTTPServer:\n    pass\n",
                ),
                ("src/runner.py", "runner", "def runner():\n    return None\n"),
                ("a/duplicate.py", "calculate_total", "def calculate_total():\n    return 1\n"),
                ("z/duplicate.py", "calculate_total", "def calculate_total():\n    return 2\n"),
            ],
        )
        self.retriever = SymbolRetriever(self.database)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_exact_qualified_symbol_beats_same_leaf_in_other_context(self):
        results = self.retriever.search(self.project_id, "Explain `Context.invoke`.")

        self.assertEqual(results[0].qualified_name, "Context.invoke")
        self.assertEqual(results[0].symbol_match_type, "exact_qualified")
        other = next(item for item in results if item.qualified_name == "Other.invoke")
        self.assertGreater(results[0].symbol_score, other.symbol_score)

    def test_multilevel_dotted_snake_case_and_camel_case(self):
        dotted = self.retriever.search(
            self.project_id, "`CliRunner.isolation.should_strip_ansi`"
        )
        snake = self.retriever.search(self.project_id, "find get_current_context")
        camel = self.retriever.search(self.project_id, "class HTTPServer")

        self.assertEqual(
            dotted[0].qualified_name, "CliRunner.isolation.should_strip_ansi"
        )
        self.assertEqual(dotted[0].symbol_match_type, "exact_qualified")
        self.assertEqual(snake[0].symbol_name, "get_current_context")
        self.assertEqual(camel[0].symbol_name, "HTTPServer")

        normalized = self.retriever.search(self.project_id, "`getCurrentContext`")
        self.assertEqual(normalized[0].symbol_name, "get_current_context")
        self.assertEqual(normalized[0].symbol_match_type, "normalized_identifier")

    def test_leaf_symbol_and_explicit_lookup_modes_remain_supported(self):
        exact = self.retriever.search(
            self.project_id, "invoke", explicit_symbol=True, match_mode="exact"
        )
        prefix = self.retriever.search(
            self.project_id, "get_current", explicit_symbol=True, match_mode="prefix"
        )
        fuzzy = self.retriever.search(
            self.project_id, "current_context", explicit_symbol=True, match_mode="fuzzy"
        )

        self.assertEqual(
            [item.qualified_name for item in exact], ["Context.invoke", "Other.invoke"]
        )
        self.assertTrue(all(item.symbol_match_type == "exact_leaf" for item in exact))
        self.assertEqual(prefix[0].symbol_name, "get_current_context")
        self.assertEqual(fuzzy[0].symbol_name, "get_current_context")

    def test_path_and_symbol_context_are_preserved(self):
        result = self.retriever.search(
            self.project_id, "`src/context.py:Context.invoke`"
        )[0]
        value = result.to_dict()

        self.assertEqual(result.path, "src/context.py")
        self.assertEqual(result.start_line, 1)
        self.assertEqual(result.end_line, 2)
        self.assertTrue(result.chunk_identity.endswith(str(result.code_chunk_id)))
        self.assertEqual(result.candidate_source, "symbol")
        self.assertIn("qualified_symbol_exact", result.match_reasons)
        self.assertEqual(value["candidate_source"], "symbol")
        self.assertEqual(value["match_reasons"], ["qualified_symbol_exact", "path_context_exact"])

    def test_plain_natural_language_common_words_do_not_trigger_symbol_search(self):
        results = self.retriever.search(
            self.project_id,
            "The context is useful and the runner returns a result option.",
        )

        self.assertEqual(results, [])

    def test_no_match_and_filters_return_safe_results(self):
        self.assertEqual(
            self.retriever.search(self.project_id, "`does_not_exist`"), []
        )
        filtered = self.retriever.search(
            self.project_id,
            "`Context.invoke`",
            path="src/server.py",
            language="python",
        )
        self.assertEqual(filtered, [])

    def test_equal_scores_have_stable_complete_tie_breakers(self):
        first = self.retriever.search(self.project_id, "`calculate_total`")
        second = self.retriever.search(self.project_id, "`calculate_total`")

        expected = ["a/duplicate.py", "z/duplicate.py"]
        self.assertEqual([item.path for item in first], expected)
        self.assertEqual([item.path for item in second], expected)
        self.assertEqual(
            [item.to_dict() for item in first], [item.to_dict() for item in second]
        )
        self.assertEqual([item.symbol_rank for item in first], [1, 2])

    def test_duplicate_hints_do_not_duplicate_candidates(self):
        results = self.retriever.search(
            self.project_id, "`Context.invoke` and Context.invoke()"
        )

        identities = [item.chunk_identity for item in results]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(results[0].qualified_name, "Context.invoke")


if __name__ == "__main__":
    unittest.main()
