from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app import local_product_smoke
from app.services.llm_client import ProviderError


class LocalProductSmokeTests(unittest.TestCase):
    def test_gate_c_emits_only_safe_provider_diagnostics(self):
        error = ProviderError(
            "provider_output_truncated",
            "The generation provider reached the configured output limit before returning final content.",
            diagnostics={
                "provider": "openai_compatible",
                "model": "configured-model",
                "finish_reason": "length",
                "content_empty": True,
                "reasoning_content_present": True,
                "response_body": "response-body",
                "reasoning_content": "reasoning-body",
                "authorization": "api-key",
            },
        )
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["local_product_smoke", "--gate-c"]),
            patch.object(local_product_smoke, "load_environment"),
            patch.object(local_product_smoke, "_create_fixture", return_value=Path("fixture")),
            patch.object(
                local_product_smoke,
                "import_local_repository",
                return_value=SimpleNamespace(snapshot=object()),
            ),
            patch.object(local_product_smoke, "_run_pipeline", side_effect=error),
            redirect_stdout(output),
        ):
            exit_code = local_product_smoke.main()

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["gate"], "C")
        self.assertEqual(payload["code"], "provider_output_truncated")
        self.assertEqual(payload["diagnostics"]["finish_reason"], "length")
        serialized = json.dumps(payload)
        for forbidden in (
            "response-body",
            "reasoning-body",
            "Authorization",
            "api-key",
        ):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
