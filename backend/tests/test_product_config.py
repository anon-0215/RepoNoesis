from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import get_embedding_settings, get_llm_settings, get_product_config_status


class ProductConfigTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
