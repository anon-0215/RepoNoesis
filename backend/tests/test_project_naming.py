from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class ProjectNamingTests(unittest.TestCase):
    def test_readme_uses_frozen_product_and_repository_names(self) -> None:
        readme = read_text("README.md")

        self.assertTrue(readme.startswith("# 源鉴 RepoNoesis\n"))
        self.assertIn(
            "《源鉴（RepoNoesis）——面向真实代码仓库的证据驱动型持续学习智能体》",
            readme,
        )
        self.assertIn("让每一次代码解释，都源于真实源码。", readme)
        self.assertIn("https://github.com/anon-0215/RepoNoesis.git", readme)
        self.assertNotIn("https://github.com/anon-0215/GitLearnAgent.git", readme)

    def test_user_facing_and_code_branding_is_reponoesis(self) -> None:
        self.assertIn(
            "<title>源鉴 RepoNoesis</title>",
            read_text("frontend/index.html"),
        )
        self.assertIn(
            'title="源鉴 RepoNoesis API"',
            read_text("backend/app/main.py"),
        )
        self.assertIn(
            '"User-Agent": "RepoNoesis/0.1"',
            read_text("backend/app/services/github_client.py"),
        )
        self.assertIn(
            '"""RepoNoesis backend package."""',
            read_text("backend/app/__init__.py"),
        )

    def test_frontend_metadata_and_export_name_use_reponoesis(self) -> None:
        package = json.loads(read_text("frontend/package.json"))
        lock = json.loads(read_text("frontend/package-lock.json"))

        self.assertEqual(package["name"], "reponoesis-frontend")
        self.assertEqual(lock["name"], "reponoesis-frontend")
        self.assertEqual(lock["packages"][""]["name"], "reponoesis-frontend")
        self.assertIn(
            "anchor.download = 'reponoesis-report.md'",
            read_text("frontend/src/App.tsx"),
        )

    def test_current_setup_text_does_not_use_old_v3_path(self) -> None:
        for relative_path in (
            ".env.example",
            "README.md",
            "docs/v3/LOCAL_PRODUCT_PHASE1.md",
        ):
            with self.subTest(path=relative_path):
                self.assertNotIn("RepoNoesis-v3", read_text(relative_path))

        github_client = read_text("backend/app/services/github_client.py")
        self.assertNotIn(r"D:\Project\GitLearnAgent\.env", github_client)

    def test_compatibility_and_history_names_are_preserved(self) -> None:
        self.assertIn("name: gitlearnagent", read_text("environment.yml"))
        self.assertIn('"gitlearnagent.sqlite"', read_text("backend/app/database.py"))
        self.assertIn("GitLearnAgent V1", read_text("docs/v3/BASELINE.md"))
        self.assertIn("# GitLearnAgent V2 开发规范", read_text("AGENTS.md"))


if __name__ == "__main__":
    unittest.main()
