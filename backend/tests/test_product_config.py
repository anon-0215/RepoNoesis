from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import (
    RepositoryConfigurationError,
    get_embedding_settings,
    get_llm_settings,
    get_product_config_status,
    get_repository_settings,
    validate_git_proxy,
)


class ProductConfigTests(unittest.TestCase):
    def setUp(self):
        self.configuration_root = tempfile.TemporaryDirectory()
        self.addCleanup(self.configuration_root.cleanup)
        root_patch = patch("app.config.PROJECT_ROOT", Path(self.configuration_root.name))
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def test_llm_configuration_requires_all_product_fields(self):
        values = {
            "LLM_PROVIDER": "openai_compatible",
            "LLM_BASE_URL": "https://provider.invalid/v1",
            "LLM_API_KEY": "secret-value-must-not-leak",
            "LLM_MODEL": "configured-model",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = get_llm_settings()
            self.assertTrue(settings.configured)
            self.assertEqual(settings.provider, "openai_compatible")
            status = get_product_config_status()
        serialized = repr(status)
        self.assertNotIn(values["LLM_API_KEY"], serialized)
        self.assertTrue(status["llm"]["api_key_configured"])
        self.assertEqual(status["llm"]["missing"], [])

    def test_missing_llm_fields_are_reported_without_secret_metadata(self):
        with patch.dict(os.environ, {}, clear=True):
            status = get_product_config_status()
        self.assertFalse(status["llm"]["ready"])
        self.assertIn("LLM_API_KEY", status["llm"]["missing"])
        self.assertNotIn("api_key_length", status["llm"])

    def test_llm_thinking_modes_are_optional_and_separate(self):
        with patch.dict(os.environ, {}, clear=True):
            defaults = get_llm_settings()
        self.assertIsNone(defaults.planner_thinking)
        self.assertIsNone(defaults.answer_thinking)

        values = {
            "LLM_PLANNER_THINKING": "disabled",
            "LLM_ANSWER_THINKING": "enabled",
        }
        with patch.dict(os.environ, values, clear=True):
            configured = get_llm_settings()
            status = get_product_config_status()
        self.assertEqual(configured.planner_thinking, "disabled")
        self.assertEqual(configured.answer_thinking, "enabled")
        self.assertEqual(status["llm"]["planner_thinking"], "disabled")
        self.assertEqual(status["llm"]["answer_thinking"], "enabled")

    def test_invalid_thinking_mode_is_rejected_without_echoing_value(self):
        with patch.dict(
            os.environ, {"LLM_PLANNER_THINKING": "private-invalid-value"}, clear=True
        ):
            with self.assertRaises(ValueError) as raised:
                get_llm_settings()
        self.assertIn("LLM_PLANNER_THINKING", str(raised.exception))
        self.assertNotIn("private-invalid-value", str(raised.exception))

    def test_embedding_product_alias_and_offline_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "bge-m3"
            model.mkdir()
            values = {
                "EMBEDDING_ENABLED": "true",
                "EMBEDDING_PROVIDER": "local_bge_m3",
                "EMBEDDING_MODEL": str(model),
                "EMBEDDING_DEVICE": "cpu",
                "EMBEDDING_OFFLINE": "true",
            }
            with patch.dict(os.environ, values, clear=True):
                settings = get_embedding_settings()
                status = get_product_config_status()
        self.assertEqual(settings.model_name_or_path, str(model))
        self.assertTrue(settings.offline)
        self.assertTrue(status["embedding"]["ready"])

    def test_git_proxy_status_is_boolean_and_settings_repr_is_redacted(self):
        marker = "proxy-credential-marker"
        value = f"https://user:{marker}@proxy.example.invalid:8443"
        with patch.dict(os.environ, {"REPONOESIS_GIT_PROXY": value}, clear=True):
            settings = get_repository_settings()
            status = get_product_config_status()
        self.assertEqual(settings.git_proxy, value)
        self.assertTrue(status["git_proxy_configured"])
        self.assertNotIn(marker, repr(settings))
        self.assertNotIn(marker, repr(status))
        self.assertNotIn("proxy.example.invalid", repr(status))

    def test_git_proxy_is_optional(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_repository_settings()
            status = get_product_config_status()
        self.assertIsNone(settings.git_proxy)
        self.assertFalse(status["git_proxy_configured"])

    def test_git_proxy_validation_accepts_only_bounded_http_urls(self):
        self.assertEqual(validate_git_proxy("http://proxy.example.invalid:8080"), "http://proxy.example.invalid:8080")
        self.assertEqual(validate_git_proxy("https://user:pass@proxy.example.invalid"), "https://user:pass@proxy.example.invalid")
        rejected = (
            "file:///tmp/proxy",
            "ssh://proxy.example.invalid",
            "socks5://proxy.example.invalid",
            "https:///missing-host",
            "https://proxy.example.invalid:70000",
            "https://proxy.example.invalid/path#fragment",
            "https://proxy.example.invalid\r\nPRIVATE",
            "https://proxy.example.invalid/" + "x" * 2048,
        )
        for value in rejected:
            with self.subTest(kind=value.split(":", 1)[0]):
                with self.assertRaises(RepositoryConfigurationError) as raised:
                    validate_git_proxy(value)
                serialized = repr(raised.exception) + str(raised.exception)
                self.assertEqual(raised.exception.code, "git_proxy_invalid")
                self.assertNotIn(value, serialized)


if __name__ == "__main__":
    unittest.main()
