import tempfile
import unittest
from pathlib import Path

from app.database import Database
from app.services.lexical_retriever import LexicalRetriever, tokenize_code_text
from tests.m1_helpers import make_project


class TokenizerTests(unittest.TestCase):
    def test_identifier_and_path_splitting(self):
        tokens = tokenize_code_text("src/AuthService.py parse_httpRequest HTTPServer")
        for expected in {
            "src",
            "authservice",
            "auth",
            "service",
            "py",
            "parse_httprequest",
            "parse",
            "http",
            "request",
            "httpserver",
            "server",
        }:
            self.assertIn(expected, tokens)

    def test_casefold_and_code_punctuation(self):
        self.assertEqual(
            tokenize_code_text("User::TOKEN(value)"),
            tokenize_code_text("user::token(VALUE)"),
        )

    def test_cjk_unigrams_bigrams_and_mixed_text(self):
        tokens = tokenize_code_text("用户验证 auth_user")
        for expected in {"用户验证", "用", "户", "验", "证", "用户", "户验", "验证", "auth_user", "auth", "user"}:
            self.assertIn(expected, tokens)


class Bm25Tests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "lexical.sqlite")
        self.project_id, _bundle = make_project(
            self.db,
            [
                (
                    "src/auth_service.py",
                    "AuthService.authenticateUser",
                    "def authenticateUser(password):\n    return verify_password(password)\n",
                ),
                (
                    "src/upload.py",
                    "upload_file",
                    "def upload_file(path):\n    return save_blob(path)\n",
                ),
                (
                    "src/z_auth.py",
                    "auth_backup",
                    "def auth_backup():\n    return 'auth'\n",
                ),
            ],
        )
        self.retriever = LexicalRetriever(self.db)

    def tearDown(self):
        self.directory.cleanup()

    def test_bm25_ranks_exact_symbol_first(self):
        results = self.retriever.search(self.project_id, "authenticateUser")
        self.assertEqual(results[0].qualified_name, "AuthService.authenticateUser")
        self.assertGreater(results[0].lexical_score, 0)
        self.assertEqual(results[0].lexical_rank, 1)

    def test_filters_path_language_and_symbol(self):
        self.assertEqual(
            [item.path for item in self.retriever.search(self.project_id, "return", path="src/upload.py")],
            ["src/upload.py"],
        )
        self.assertTrue(self.retriever.search(self.project_id, "auth", language="PYTHON"))
        self.assertEqual(
            self.retriever.search(self.project_id, "auth", language="javascript"),
            [],
        )
        self.assertEqual(
            [item.qualified_name for item in self.retriever.search(
                self.project_id,
                "password",
                symbol="AuthService.authenticateUser",
            )],
            ["AuthService.authenticateUser"],
        )

    def test_empty_query_and_empty_index(self):
        self.assertEqual(self.retriever.search(self.project_id, "!!!"), [])
        empty_project, _ = make_project(self.db, [])
        self.assertEqual(self.retriever.search(empty_project, "auth"), [])

    def test_equal_scores_have_stable_order(self):
        first = self.retriever.search(self.project_id, "auth", top_k=20)
        second = self.retriever.search(self.project_id, "auth", top_k=20)
        self.assertEqual(
            [(item.path, item.code_chunk_id) for item in first],
            [(item.path, item.code_chunk_id) for item in second],
        )


if __name__ == "__main__":
    unittest.main()
