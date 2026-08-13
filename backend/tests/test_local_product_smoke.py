from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app import local_product_smoke
from app.config import EmbeddingSettings
from app.services.llm_client import ProviderError
from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder, SmokeGateError


class LocalProductSmokeTests(unittest.TestCase):
    def _configured_embedding(self):
        return EmbeddingSettings(
            enabled=True,
            model_name_or_path="local-model",
            device="cpu",
            batch_size=1,
            max_length=128,
            normalize=True,
            cache_dir=Path("unused-cache"),
            query_prefix="",
            document_prefix="",
            provider="local_bge_m3",
            offline=True,
        )

    def _pipeline_patches(self, result, *, database_paths=None):
        database = MagicMock()
        database.create_project.return_value = "project-1"
        database.get_bundle.return_value = {"project": {"id": "project-1"}}
        def create_database(path):
            if database_paths is not None:
                database_paths.append(Path(path))
                Path(path).touch()
            return database
        chunk = SimpleNamespace(to_dict=lambda: {"id": 1})
        snapshot = SimpleNamespace(
            to_dict=lambda: {"repo": "smoke-fixture"},
            files=[],
            repository_revision="revision",
            repo="smoke-fixture",
            repo_url="local",
        )
        patches = (
            patch.object(local_product_smoke, "get_embedding_settings", return_value=self._configured_embedding()),
            patch.object(local_product_smoke, "EmbeddingService", return_value=MagicMock()),
            patch.object(local_product_smoke, "Database", side_effect=create_database),
            patch.object(local_product_smoke, "analyze_snapshot", return_value={"files": []}),
            patch.object(
                local_product_smoke,
                "extract_python_code_chunks_from_files",
                return_value=SimpleNamespace(chunks=[chunk]),
            ),
            patch.object(local_product_smoke, "build_learning_path", return_value={}),
            patch.object(local_product_smoke, "index_project_relations", return_value=MagicMock()),
            patch.object(local_product_smoke, "EmbeddingIndexer", return_value=MagicMock()),
            patch.object(local_product_smoke, "run_bounded_agent", return_value=result),
        )
        return snapshot, patches

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

    def test_embedding_configuration_error_has_stable_code(self):
        recorder = SmokeDiagnosticsRecorder()
        settings = self._configured_embedding()
        settings = EmbeddingSettings(**{**settings.__dict__, "offline": False})
        with patch.object(local_product_smoke, "get_embedding_settings", return_value=settings):
            with self.assertRaises(SmokeGateError) as raised:
                local_product_smoke._run_pipeline(
                    object(), require_provider=False, gate="C", diagnostics_recorder=recorder
                )
        self.assertEqual(raised.exception.code, "smoke_embedding_configuration_incomplete")
        self.assertEqual(raised.exception.stage, "embedding_preflight")

    def test_no_python_chunks_has_stable_code(self):
        recorder = SmokeDiagnosticsRecorder()
        database = MagicMock()
        database.create_project.return_value = "project-1"
        snapshot = SimpleNamespace(
            to_dict=lambda: {"repo": "smoke-fixture"},
            files=[],
            repository_revision="revision",
        )
        with (
            patch.object(local_product_smoke, "get_embedding_settings", return_value=self._configured_embedding()),
            patch.object(local_product_smoke, "EmbeddingService", return_value=MagicMock()),
            patch.object(local_product_smoke, "Database", return_value=database),
            patch.object(local_product_smoke, "analyze_snapshot", return_value={"files": []}),
            patch.object(
                local_product_smoke,
                "extract_python_code_chunks_from_files",
                return_value=SimpleNamespace(chunks=[]),
            ),
        ):
            with self.assertRaises(SmokeGateError) as raised:
                local_product_smoke._run_pipeline(
                    snapshot, require_provider=False, gate="C", diagnostics_recorder=recorder
                )
        self.assertEqual(raised.exception.code, "smoke_no_python_chunks")
        self.assertEqual(raised.exception.stage, "chunk_extraction")

    def test_gate_assertions_have_distinct_stable_codes(self):
        cases = (
            (
                "missing_evidence",
                {"citations": [], "evidence": [], "agent_mode": "bounded", "answer_mode": "llm_grounded"},
                "smoke_validated_evidence_missing",
            ),
            (
                "grounding_failed",
                {
                    "citations": [{}],
                    "evidence": [{}],
                    "agent_mode": "bounded",
                    "answer_mode": "deterministic",
                    "agent_status": "completed",
                },
                "smoke_provider_grounding_failed",
            ),
        )
        for label, result, expected_code in cases:
            with self.subTest(label=label):
                recorder = SmokeDiagnosticsRecorder()
                snapshot, patches = self._pipeline_patches(result)
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(local_product_smoke, "LLMClient", return_value=MagicMock())
                    )
                    for context in patches:
                        stack.enter_context(context)
                    with self.assertRaises(SmokeGateError) as raised:
                        local_product_smoke._run_pipeline(
                            snapshot,
                            require_provider=True,
                            gate="C",
                            diagnostics_recorder=recorder,
                        )
                self.assertEqual(raised.exception.code, expected_code)
                self.assertEqual(raised.exception.stage, "gate_assertion")

    def test_generic_failure_reports_only_stable_stage_and_type(self):
        def fail_pipeline(*_args, **kwargs):
            kwargs["diagnostics_recorder"].enter_stage("embedding_index")
            raise RuntimeError("sensitive exception detail")

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
            patch.object(local_product_smoke, "_run_pipeline", side_effect=fail_pipeline),
            redirect_stdout(output),
        ):
            exit_code = local_product_smoke.main()
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(payload["code"], "smoke_stage_failed")
        self.assertEqual(payload["stage"], "embedding_index")
        self.assertEqual(payload["exception_type"], "RuntimeError")
        self.assertNotIn("sensitive exception detail", json.dumps(payload))

    def test_temporary_fixture_is_cleaned_after_success_and_failure(self):
        observed = []

        def create_fixture(parent):
            observed.append(parent)
            return parent / "fixture"

        for failure in (False, True):
            output = io.StringIO()
            effect = RuntimeError("safe-test") if failure else {"status": "pass"}
            with (
                patch.object(sys, "argv", ["local_product_smoke", "--gate-c"]),
                patch.object(local_product_smoke, "load_environment"),
                patch.object(local_product_smoke, "_create_fixture", side_effect=create_fixture),
                patch.object(
                    local_product_smoke,
                    "import_local_repository",
                    return_value=SimpleNamespace(snapshot=object()),
                ),
                patch.object(local_product_smoke, "_run_pipeline", side_effect=effect if failure else None, return_value=effect if not failure else None),
                redirect_stdout(output),
            ):
                local_product_smoke.main()
            self.assertFalse(observed[-1].exists())

    def test_temporary_sqlite_is_cleaned_after_success_and_failure(self):
        success = {
            "citations": [{}],
            "evidence": [{}],
            "agent_mode": "bounded",
            "answer_mode": "llm_grounded",
            "agent_status": "completed",
        }
        failure = {
            **success,
            "answer_mode": "deterministic",
        }
        for label, result in (("success", success), ("failure", failure)):
            with self.subTest(label=label):
                database_paths = []
                recorder = SmokeDiagnosticsRecorder()
                snapshot, patches = self._pipeline_patches(
                    result, database_paths=database_paths
                )
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            local_product_smoke, "LLMClient", return_value=MagicMock()
                        )
                    )
                    for context in patches:
                        stack.enter_context(context)
                    if label == "failure":
                        with self.assertRaises(SmokeGateError):
                            local_product_smoke._run_pipeline(
                                snapshot,
                                require_provider=True,
                                gate="C",
                                diagnostics_recorder=recorder,
                            )
                    else:
                        local_product_smoke._run_pipeline(
                            snapshot,
                            require_provider=True,
                            gate="C",
                            diagnostics_recorder=recorder,
                        )
                self.assertEqual(len(database_paths), 1)
                self.assertFalse(database_paths[0].parent.exists())


if __name__ == "__main__":
    unittest.main()
