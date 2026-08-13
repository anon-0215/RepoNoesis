from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.config import EmbeddingSettings, RepositorySettings
from app.database import Database, SCHEMA_VERSION
from app.models import RepoFile, RepositorySnapshot
from app.services.embedding_service import EmbeddingService
from app.services.learning_continuity import (
    CONTINUITY_MAPPING_CONFIG_IDENTITY,
    LearningContinuityService,
)
from app.services.learning_contracts import (
    CreateGoalRequest,
    CreatePlanRequest,
    CreateTaskRequest,
    PlanStepInput,
    RubricCriterionInput,
    SelfReportRequest,
    SubmitAttemptRequest,
    TargetSpec,
)
from app.services.learning_service import LearningService
from app.services.repository_import import ImportedRepository
from app.services.workspace_update import WorkspaceUpdateService
from tests.m4_helpers import FakeEvaluator
from tests.p22_helpers import build_fixture_snapshot


class _Backend:
    def load_model(self, *_args, **_kwargs): return None
    def encode(self, texts, batch_size, normalize):
        del batch_size, normalize
        return [[1.0, 0.0] for _ in texts]
    def get_embedding_dimension(self): return 2
    def get_model_revision(self): return None
    def unload_model(self): return None


class LearningContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.root = root / "repo"
        self.root.mkdir()
        self.database = Database(root / "continuity.sqlite")
        settings = EmbeddingSettings(
            enabled=True, model_name_or_path="fake-bge-m3", device="cpu",
            batch_size=4, max_length=128, normalize=True, cache_dir=root / "cache",
            query_prefix="", document_prefix="", model_revision="fake-revision",
            provider="local_bge_m3", offline=True,
        )
        self.embedding = EmbeddingService(
            settings, backend_factory=_Backend, cuda_available=lambda: False
        )
        self.current: ImportedRepository | None = None
        self.update = WorkspaceUpdateService(
            self.database,
            RepositorySettings(root / "runtime"),
            self.embedding,
            importer=lambda *_args: self.current,
        )

    def _imported(self, revision: str, files: dict[str, str]) -> ImportedRepository:
        snapshot = RepositorySnapshot(
            repo_url=str(self.root), owner="local", repo="fixture", default_branch="main",
            files=[
                RepoFile(
                    path=path, size=len(content.encode()), content=content,
                    extension=Path(path).suffix,
                )
                for path, content in files.items()
            ],
            repository_revision=revision, source_type="local",
            source_location=str(self.root), source_identity=f"identity-{revision}",
        )
        return ImportedRepository(snapshot, snapshot.source_identity, self.root)

    def _seed(self, files: dict[str, str]) -> tuple[str, str, LearningService]:
        self.current = self._imported("a" * 40, files)
        project_id = self.database.create_project(self.current.snapshot.to_dict())
        build_fixture_snapshot(self.database, self.embedding, project_id, self.current.snapshot)
        workspace_id = self.database.get_workspace_for_project(project_id)["id"]
        return project_id, workspace_id, LearningService(self.database)

    def _master_symbol(self, learning: LearningService, project_id: str, path: str, name: str) -> str:
        goal = learning.create_goal(
            project_id,
            CreateGoalRequest(
                goal_text=f"理解 {name}", goal_type="symbol_understanding",
                idempotency_key=f"goal-{name}-0001",
            ),
        )
        plan = learning.create_plan(
            project_id,
            CreatePlanRequest(
                goal_id=goal["goal_id"], expected_current_version=0,
                idempotency_key=f"plan-{name}-0001",
                steps=[
                    PlanStepInput(
                        objective="第一次验证", action_type="explain_symbol",
                        completion_requirement="通过 Evidence 任务",
                        target=TargetSpec(target_type="symbol", path=path, qualified_name=name),
                    ),
                    PlanStepInput(
                        objective="第二次验证", action_type="checkpoint",
                        completion_requirement="通过另一项 Evidence 任务",
                        target=TargetSpec(target_type="symbol", path=path, qualified_name=name),
                    ),
                ],
            ),
        )
        for index in range(1, 3):
            if index > 1:
                plan = learning.get_current_plan(project_id, goal["goal_id"])
            step = next(
                item for item in plan["steps"]
                if item["status"] in {"active", "pending", "needs_review"}
            )
            task = learning.create_task(
                project_id,
                CreateTaskRequest(
                    plan_id=plan["plan_id"], plan_version=plan["version"],
                    step_id=step["step_id"], task_type="explain_symbol",
                    prompt_text="解释目标并引用 Evidence。",
                    rubric=[RubricCriterionInput(
                        criterion_id="source_fact", criterion_type="source_fact",
                        weight=1.0, expected_claim="说明目标职责", critical=True,
                    )],
                    idempotency_key=f"task-{name}-{index:04d}",
                ),
            )
            learning.submit_attempt(
                project_id, task["task_id"],
                SubmitAttemptRequest(
                    answer_text="valid", idempotency_key=f"attempt-{name}-{index:04d}"
                ),
                evaluator=FakeEvaluator("pass"),
            )
        state = next(
            item for item in learning.get_states(project_id)
            if item["path"] == path and item["qualified_name"] == name
        )
        self.assertEqual(state["mastery_status"], "mastered")
        return state["target_id"]

    def _refresh(self, workspace_id: str, files: dict[str, str]) -> tuple[str, dict]:
        self.current = self._imported("b" * 40, files)
        run = self.update.start_refresh(workspace_id)
        completed = self.update.execute_run(workspace_id, run["run_id"])
        return completed["project_id"], completed

    def _introduce_symbol(
        self, learning: LearningService, project_id: str, path: str, name: str, key: str
    ) -> None:
        learning.submit_self_report(
            project_id,
            SelfReportRequest(
                target=TargetSpec(target_type="symbol", path=path, qualified_name=name),
                report_text="seen", idempotency_key=key,
            ),
        )

    def _introduce_file(
        self, learning: LearningService, project_id: str, path: str, key: str
    ) -> None:
        learning.submit_self_report(
            project_id,
            SelfReportRequest(
                target=TargetSpec(target_type="file", path=path),
                report_text="seen", idempotency_key=key,
            ),
        )

    def test_schema_v11_is_additive_and_keeps_immutable_event_triggers(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 11)
        with self.database.connect() as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            self.assertTrue({
                "learning_continuity_transitions",
                "learning_continuity_mappings",
                "learning_continuity_goal_lineage",
            }.issubset(tables))
            triggers = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )}
            self.assertIn("trg_learning_events_no_update", triggers)
            self.assertIn("trg_learning_events_no_delete", triggers)

    def test_v10_with_nonempty_m4_data_migrates_idempotently_without_history_changes(self) -> None:
        project_id, _workspace_id, learning = self._seed(
            {"stable.py": "def stable():\n    return 1\n"}
        )
        self._master_symbol(learning, project_id, "stable.py", "stable")
        before = self.database.learning_record_counts(project_id)
        with self.database.connect() as conn:
            conn.execute("DROP TABLE learning_continuity_goal_lineage")
            conn.execute("DROP TABLE learning_continuity_mappings")
            conn.execute("DROP TABLE learning_continuity_transitions")
            conn.execute("UPDATE schema_versions SET version=10 WHERE key='database'")
        migrated = Database(self.database.path)
        Database(self.database.path)
        self.assertEqual(migrated.learning_record_counts(project_id), before)
        with migrated.connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT version FROM schema_versions WHERE key='database'"
            ).fetchone()[0], 11)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_continuity_transitions"
            ).fetchone()[0], 0)

    def test_v11_migration_failure_rolls_back_schema_and_version(self) -> None:
        database = Database(Path(self.directory.name) / "rollback.sqlite")
        with database.connect() as conn:
            conn.execute("DROP TABLE learning_continuity_goal_lineage")
            conn.execute("DROP TABLE learning_continuity_mappings")
            conn.execute("DROP TABLE learning_continuity_transitions")
            conn.execute("UPDATE schema_versions SET version=10 WHERE key='database'")
        original = Database._migrate_learning_continuity_schema

        def fail(instance, conn):
            original(instance, conn)
            raise RuntimeError("forced v11 migration failure")

        with patch.object(Database, "_migrate_learning_continuity_schema", new=fail):
            with self.assertRaises(RuntimeError):
                Database(database.path)
        conn = sqlite3.connect(database.path)
        try:
            self.assertEqual(conn.execute(
                "SELECT version FROM schema_versions WHERE key='database'"
            ).fetchone()[0], 10)
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            self.assertNotIn("learning_continuity_transitions", tables)
        finally:
            conn.close()

    def test_activation_creates_one_bound_pending_transition(self) -> None:
        old_project, workspace_id, _learning = self._seed(
            {"stable.py": "def stable():\n    return 1\n"}
        )
        new_project, completed = self._refresh(
            workspace_id, {"stable.py": "def stable():\n    return 1\n"}
        )
        transition = LearningContinuityService(self.database).get_current(workspace_id)
        self.assertEqual(transition["source_project_id"], old_project)
        self.assertEqual(transition["target_project_id"], new_project)
        self.assertEqual(transition["activation_version"], 2)
        self.assertEqual(transition["mapping_config_identity"], CONTINUITY_MAPPING_CONFIG_IDENTITY)
        self.assertEqual(completed["status"], "succeeded")

    def test_mapping_and_mastery_rules_are_conservative_and_auditable(self) -> None:
        old_project, workspace_id, learning = self._seed({
            "stable.py": "def stable():\n    return 1\n",
            "modified.py": "def modified():\n    return 1\n",
            "deleted.py": "def deleted():\n    return 1\n",
            "old_name.py": "def moved():\n    return 1\n",
            "ambiguous.py": "def duplicate():\n    return 1\n",
        })
        stable_target = self._master_symbol(learning, old_project, "stable.py", "stable")
        for path, name in (
            ("modified.py", "modified"),
            ("deleted.py", "deleted"),
            ("ambiguous.py", "duplicate"),
        ):
            self._introduce_symbol(
                learning, old_project, path, name, f"report-{name}-0001"
            )
        self._introduce_file(learning, old_project, "old_name.py", "file-rename-old")
        new_project, _completed = self._refresh(workspace_id, {
            "stable.py": "def stable():\n    return 1\n",
            "modified.py": "def modified():\n    return 2\n",
            "new_name.py": "def moved():\n    return 1\n",
            "copy_one.py": "def duplicate():\n    return 1\n",
            "copy_two.py": "def duplicate():\n    return 1\n",
        })
        service = LearningContinuityService(self.database)
        transition = service.get_current(workspace_id)
        self.assertEqual(transition["status"], "succeeded")
        mappings = service.get_impacts(workspace_id, transition["transition_id"])["items"]
        by_source = {item["source_path"]: item for item in mappings}
        self.assertEqual(by_source["stable.py"]["mapping_status"], "unchanged_exact")
        self.assertEqual(by_source["modified.py"]["mapping_status"], "modified")
        self.assertEqual(by_source["deleted.py"]["mapping_status"], "deleted")
        self.assertEqual(by_source["old_name.py"]["mapping_status"], "renamed_exact")
        self.assertEqual(by_source["ambiguous.py"]["mapping_status"], "ambiguous")

        target_states = learning.get_states(new_project)
        by_path = {item["path"]: item for item in target_states}
        self.assertEqual(by_path["stable.py"]["mastery_status"], "mastered")
        self.assertEqual(by_path["modified.py"]["mastery_status"], "needs_review")
        self.assertNotIn("deleted.py", by_path)
        self.assertNotIn("ambiguous.py", by_path)
        self.assertEqual(by_path["stable.py"]["review_reason"], "")
        with self.database.connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_attempts WHERE project_id=?", (new_project,)
            ).fetchone()[0], 0)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_evaluations e JOIN learning_attempts a "
                "ON a.attempt_id=e.attempt_id WHERE a.project_id=?", (new_project,)
            ).fetchone()[0], 0)
            events = conn.execute(
                "SELECT event_type, provenance, attempt_id, evaluation_id FROM learning_events "
                "WHERE project_id=?", (new_project,)
            ).fetchall()
            self.assertTrue(events)
            self.assertTrue(all(row["event_type"] == "continuity_state_derived" for row in events))
            self.assertTrue(all(row["provenance"] == "revision_continuity" for row in events))
            self.assertTrue(all(row["attempt_id"] is None and row["evaluation_id"] is None for row in events))
        context = learning.get_learning_context(new_project)
        self.assertEqual(context["project_binding"]["project_id"], new_project)
        self.assertGreater(context["metrics"]["needs_review_count"], 0)
        self.assertEqual(context["recommended_explanation_depth"], "foundational_review")
        self.assertEqual(stable_target, by_source["stable.py"]["source_target_id"])
        self._master_symbol(learning, new_project, "modified.py", "modified")
        remastered = {
            item["path"]: item for item in learning.get_states(new_project)
        }
        self.assertEqual(remastered["modified.py"]["mastery_status"], "mastered")
        with self.database.connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_attempts WHERE project_id=?", (new_project,)
            ).fetchone()[0], 2)

    def test_line_shift_is_exact_but_symbol_rename_requires_review(self) -> None:
        old_project, workspace_id, learning = self._seed({
            "app.py": "def stable():\n    return 1\n\ndef old_name():\n    return 2\n",
        })
        self._introduce_symbol(learning, old_project, "app.py", "stable", "line-shift-stable")
        self._introduce_symbol(learning, old_project, "app.py", "old_name", "symbol-rename-old")
        self._refresh(workspace_id, {
            "app.py": "# inserted line\ndef stable():\n    return 1\n\ndef new_name():\n    return 2\n",
        })
        service = LearningContinuityService(self.database)
        transition = service.get_current(workspace_id)
        impacts = service.get_impacts(workspace_id, transition["transition_id"])["items"]
        by_name = {item["source_qualified_name"]: item for item in impacts}
        self.assertEqual(by_name["stable"]["mapping_status"], "unchanged_exact")
        self.assertEqual(by_name["old_name"]["mapping_status"], "modified")
        self.assertEqual(by_name["old_name"]["target_qualified_name"], "new_name")

    def test_chunker_identity_change_is_incompatible_and_never_inherits(self) -> None:
        old_project, workspace_id, learning = self._seed(
            {"app.py": "def stable():\n    return 1\n"}
        )
        self._master_symbol(learning, old_project, "app.py", "stable")
        with self.database.connect() as conn:
            conn.execute(
                "UPDATE workspace_revisions SET chunker_version='obsolete' WHERE project_id=?",
                (old_project,),
            )
        new_project, _completed = self._refresh(
            workspace_id, {"app.py": "def stable():\n    return 1\n"}
        )
        service = LearningContinuityService(self.database)
        transition = service.get_current(workspace_id)
        impacts = service.get_impacts(workspace_id, transition["transition_id"])["items"]
        self.assertTrue(impacts)
        self.assertTrue(all(item["mapping_status"] == "incompatible" for item in impacts))
        self.assertEqual(learning.get_states(new_project), [])

    def test_many_old_targets_cannot_claim_one_new_target(self) -> None:
        old_project, workspace_id, learning = self._seed({
            "one.py": "def duplicate():\n    return 1\n",
            "two.py": "def duplicate():\n    return 1\n",
        })
        self._introduce_symbol(learning, old_project, "one.py", "duplicate", "duplicate-one")
        self._introduce_symbol(learning, old_project, "two.py", "duplicate", "duplicate-two")
        new_project, _completed = self._refresh(
            workspace_id, {"only.py": "def duplicate():\n    return 1\n"}
        )
        service = LearningContinuityService(self.database)
        transition = service.get_current(workspace_id)
        impacts = service.get_impacts(workspace_id, transition["transition_id"])["items"]
        self.assertEqual([item["source_path"] for item in impacts], ["one.py", "two.py"])
        self.assertTrue(all(item["mapping_status"] == "ambiguous" for item in impacts))
        self.assertEqual(learning.get_states(new_project), [])

    def test_duplicate_concurrent_execute_and_restart_are_idempotent(self) -> None:
        old_project, workspace_id, learning = self._seed(
            {"stable.py": "def stable():\n    return 1\n"}
        )
        self._master_symbol(learning, old_project, "stable.py", "stable")
        new_project, _completed = self._refresh(
            workspace_id, {"stable.py": "def stable():\n    return 1\n"}
        )
        service = LearningContinuityService(self.database)
        transition = service.get_current(workspace_id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda _item: service.execute(workspace_id, transition["transition_id"]),
                range(2),
            ))
        self.assertTrue(all(item["status"] == "succeeded" for item in results))
        restarted = LearningContinuityService(Database(self.database.path))
        after = restarted.get_current(workspace_id)
        self.assertEqual(after["transition_id"], transition["transition_id"])
        with self.database.connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_events WHERE project_id=?", (new_project,)
            ).fetchone()[0], 1)

    def test_pending_or_running_transition_is_failed_on_restart(self) -> None:
        old_project, workspace_id, learning = self._seed(
            {"stable.py": "def stable():\n    return 1\n"}
        )
        self._master_symbol(learning, old_project, "stable.py", "stable")
        new_project, _completed = self._refresh(
            workspace_id, {"stable.py": "def stable():\n    return 1\n"}
        )
        transition = LearningContinuityService(self.database).get_current(workspace_id)
        restarted = LearningContinuityService(Database(self.database.path))
        for status in ("pending", "running"):
            with self.subTest(status=status):
                with self.database.connect() as conn:
                    conn.execute(
                        "UPDATE learning_continuity_transitions SET status=? WHERE id=?",
                        (status, transition["transition_id"]),
                    )
                self.assertEqual(restarted.recover_interrupted(), 1)
                failed = restarted.get_current(workspace_id)
                self.assertEqual(failed["error_code"], "continuity_interrupted")

    def test_stale_transition_cannot_publish_after_a_newer_activation(self) -> None:
        old_project, workspace_id, learning = self._seed(
            {"stable.py": "def stable():\n    return 1\n"}
        )
        self._master_symbol(learning, old_project, "stable.py", "stable")
        self.update.continuity_fail_before_publish = True
        project_b, _completed_b = self._refresh(
            workspace_id, {"stable.py": "def stable():\n    return 1\n"}
        )
        service = LearningContinuityService(self.database)
        old_transition = service.get_current(workspace_id)
        self.assertEqual(old_transition["status"], "failed")

        self.update.continuity_fail_before_publish = False
        self.current = self._imported(
            "c" * 40, {"stable.py": "def stable():\n    return 2\n"}
        )
        run = self.update.start_refresh(workspace_id)
        completed_c = self.update.execute_run(workspace_id, run["run_id"])
        self.assertNotEqual(completed_c["project_id"], project_b)

        service.retry(workspace_id, old_transition["transition_id"])
        stale = service.execute(workspace_id, old_transition["transition_id"])
        self.assertEqual(stale["status"], "failed")
        self.assertEqual(stale["error_code"], "continuity_publish_failed")
        self.assertEqual(self.database.get_workspace_record(workspace_id)["active_project_id"], completed_c["project_id"])

    def test_failure_publishes_nothing_and_explicit_retry_recovers(self) -> None:
        old_project, workspace_id, learning = self._seed(
            {"stable.py": "def stable():\n    return 1\n"}
        )
        self._master_symbol(learning, old_project, "stable.py", "stable")
        self.current = self._imported(
            "b" * 40, {"stable.py": "def stable():\n    return 1\n"}
        )
        self.update.continuity_fail_before_publish = True
        run = self.update.start_refresh(workspace_id)
        completed = self.update.execute_run(workspace_id, run["run_id"])
        new_project = completed["project_id"]
        service = LearningContinuityService(self.database)
        transition = service.get_current(workspace_id)
        self.assertEqual(transition["status"], "failed")
        with self.database.connect() as conn:
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_continuity_mappings WHERE transition_id=?",
                (transition["transition_id"],),
            ).fetchone()[0], 0)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM learning_events WHERE project_id=?", (new_project,)
            ).fetchone()[0], 0)
        self.assertEqual(learning.get_learning_context(new_project)["learning_mode"], "disabled")
        service.retry(workspace_id, transition["transition_id"])
        succeeded = service.execute(workspace_id, transition["transition_id"])
        self.assertEqual(succeeded["status"], "succeeded")
        self.assertEqual(self.database.get_workspace_record(workspace_id)["active_project_id"], new_project)


if __name__ == "__main__":
    unittest.main()
