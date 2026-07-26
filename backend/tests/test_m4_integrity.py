from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from app.database import Database, SCHEMA_VERSION
from app.services.learning_contracts import (
    CreateGoalRequest,
    CreatePlanRequest,
    PlanStepInput,
    SelfReportRequest,
    SubmitAttemptRequest,
    TargetSpec,
)
from app.services.learning_service import LearningConflict, LearningError, LearningService
from app.services.learning_validation import LearningStateValidator
from tests.m3_helpers import make_relation_project
from tests.m4_helpers import FakeEvaluator, create_goal_plan_task


class M4IntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "m4-integrity.sqlite"
        self.db = Database(self.path)
        self.project_id, _bundle = make_relation_project(
            self.db,
            {
                "app.py": "def target():\n    return 1\n",
                "other.py": "def other():\n    return 2\n",
            },
        )
        self.service = LearningService(self.db)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_v5_to_v6_migration_preserves_project_chunks_and_relations(self):
        relation_count = len(self.db.get_relations(self.project_id, "revision-m3"))
        with self.db.connect() as conn:
            conn.execute("UPDATE schema_versions SET version=5 WHERE key='database'")
        reopened = Database(self.path)
        self.assertEqual(reopened.get_project(self.project_id)["repo"], "reponoesis-m3-fixture")
        self.assertEqual(len(reopened.get_code_chunks(self.project_id)), 2)
        self.assertEqual(len(reopened.get_relations(self.project_id, "revision-m3")), relation_count)
        with reopened.connect() as conn:
            self.assertEqual(conn.execute("SELECT version FROM schema_versions WHERE key='database'").fetchone()[0], 6)
            self.assertIn("learning_events", {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")})

    def test_migration_failure_keeps_previous_schema_version(self):
        with self.db.connect() as conn:
            conn.execute("UPDATE schema_versions SET version=5 WHERE key='database'")
        with patch.object(Database, "_migrate_schema", side_effect=RuntimeError("forced")):
            with self.assertRaises(RuntimeError):
                Database(self.path)
        raw = sqlite3.connect(self.path)
        try:
            self.assertEqual(raw.execute("SELECT version FROM schema_versions WHERE key='database'").fetchone()[0], 5)
        finally:
            raw.close()

    def test_attempt_event_state_plan_failure_rolls_back_everything(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        with patch.object(
            self.service,
            "_rebuild_target_state_tx",
            side_effect=RuntimeError("forced projection failure"),
        ):
            with self.assertRaises(RuntimeError):
                self.service.submit_attempt(
                    self.project_id,
                    task["task_id"],
                    SubmitAttemptRequest(answer_text="valid", idempotency_key="rollback-attempt"),
                    evaluator=FakeEvaluator("pass"),
                )
        with self.db.connect() as conn:
            for table in ("learning_attempts", "learning_evaluations", "learning_events", "learner_target_states"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_plans").fetchone()[0], 1)

    def test_concurrent_duplicate_attempt_produces_one_event(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        request = SubmitAttemptRequest(answer_text="valid", idempotency_key="concurrent-attempt")

        def submit() -> str:
            service = LearningService(Database(self.path))
            return service.submit_attempt(
                self.project_id, task["task_id"], request, evaluator=FakeEvaluator("pass")
            )["attempt_id"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            ids = list(pool.map(lambda _value: submit(), range(2)))
        self.assertEqual(ids[0], ids[1])
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_attempts").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0], 1)

    def test_stale_plan_task_cannot_submit_after_other_attempt_adapts(self):
        _goal, plan, first_task = create_goal_plan_task(self.service, self.project_id)
        second_step = plan["steps"][1]
        from app.services.learning_contracts import CreateTaskRequest, RubricCriterionInput
        second_task = self.service.create_task(
            self.project_id,
            CreateTaskRequest(
                plan_id=plan["plan_id"],
                plan_version=plan["version"],
                step_id=second_step["step_id"],
                task_type="explain_symbol",
                prompt_text="second",
                rubric=[RubricCriterionInput(
                    criterion_id="source_fact", criterion_type="source_fact",
                    weight=1.0, expected_claim="target fact", critical=True,
                )],
                idempotency_key="parallel-task-key",
            ),
        )
        self.service.submit_attempt(
            self.project_id,
            first_task["task_id"],
            SubmitAttemptRequest(answer_text="valid", idempotency_key="first-adapts-plan"),
            evaluator=FakeEvaluator("pass"),
        )
        with self.assertRaises(LearningConflict):
            self.service.submit_attempt(
                self.project_id,
                second_task["task_id"],
                SubmitAttemptRequest(answer_text="valid", idempotency_key="stale-plan-attempt"),
                evaluator=FakeEvaluator("pass"),
            )

    def test_revision_content_change_requires_review_and_inserts_review_step(self):
        _goal, plan, task = create_goal_plan_task(self.service, self.project_id)
        self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="valid", idempotency_key="before-change"),
            evaluator=FakeEvaluator("pass"),
        )
        changed = "def target():\n    return 99\n"
        changed_hash = hashlib.sha256(changed.encode("utf-8")).hexdigest()
        with self.db.connect() as conn:
            conn.execute("UPDATE projects SET repository_revision='revision-new' WHERE id=?", (self.project_id,))
            conn.execute("UPDATE repo_files SET content=? WHERE project_id=? AND path='app.py'", (changed, self.project_id))
            conn.execute("UPDATE code_chunks SET repository_revision='revision-new', content=?, content_hash=? WHERE project_id=? AND qualified_name='target'", (changed, changed_hash, self.project_id))
        result = self.service.revalidate_project(self.project_id)
        state = next(item for item in result["states"] if item["qualified_name"] == "target")
        self.assertEqual(state["availability"], "changed")
        self.assertEqual(state["mastery_status"], "needs_review")
        current = self.service.get_current_plan(self.project_id)
        self.assertGreater(current["version"], plan["version"])
        self.assertEqual(current["adaptation_reason"], "revision_requires_target_review")
        self.assertTrue(any(step["action_type"] == "review" for step in current["steps"]))

    def test_deleted_target_is_missing_and_history_remains(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        attempt = self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="valid", idempotency_key="before-delete"),
            evaluator=FakeEvaluator("pass"),
        )
        with self.db.connect() as conn:
            conn.execute("UPDATE projects SET repository_revision='revision-delete' WHERE id=?", (self.project_id,))
            conn.execute("DELETE FROM code_chunks WHERE project_id=? AND qualified_name='target'", (self.project_id,))
            conn.execute("DELETE FROM repo_files WHERE project_id=? AND path='app.py'", (self.project_id,))
        result = self.service.revalidate_project(self.project_id)
        state = next(item for item in result["states"] if item["target_id"] == task["target_id"])
        self.assertEqual(state["availability"], "missing")
        self.assertEqual(state["mastery_status"], "needs_review")
        with self.db.connect() as conn:
            self.assertIsNotNone(conn.execute("SELECT event_id FROM learning_events WHERE event_id=?", (attempt["event_id"],)).fetchone())

    def test_ambiguous_same_hash_target_is_not_auto_mapped(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM code_chunks WHERE project_id=? AND qualified_name='target'", (self.project_id,)).fetchone()
            conn.execute("UPDATE projects SET repository_revision='revision-ambiguous' WHERE id=?", (self.project_id,))
            conn.execute("UPDATE code_chunks SET repository_revision='revision-ambiguous', path='renamed_a.py', qualified_name='a.target' WHERE id=?", (row["id"],))
            conn.execute(
                """
                INSERT INTO code_chunks (
                    project_id, repository_revision, language, path, chunk_type,
                    symbol_name, qualified_name, parent_symbol, start_line,
                    end_line, content, content_hash
                ) VALUES (?, 'revision-ambiguous', 'python', 'renamed_b.py',
                    'function', 'target', 'b.target', '', 1, 2, ?, ?)
                """,
                (self.project_id, row["content"], row["content_hash"]),
            )
        result = self.service.revalidate_project(self.project_id)
        state = next(item for item in result["states"] if item["target_id"] == task["target_id"])
        self.assertEqual(state["availability"], "ambiguous")
        self.assertEqual(state["mastery_status"], "needs_review")

    def test_context_enforces_item_and_byte_limits_without_answer_history(self):
        for index in range(20):
            self.service.submit_self_report(
                self.project_id,
                SelfReportRequest(
                    target=TargetSpec(target_type="bounded_concept", concept=f"concept-{index}"),
                    report_text="self report " + ("x" * 200),
                    idempotency_key=f"context-report-{index:02d}",
                ),
            )
        self.service.create_goal(
            self.project_id,
            CreateGoalRequest(
                goal_text="bounded context", goal_type="custom_bounded",
                idempotency_key="context-goal-key",
            ),
        )
        context = self.service.get_learning_context(self.project_id, max_items=999)
        self.assertLessEqual(len(context["target_states"]), 16)
        self.assertLessEqual(len(str(context).encode("utf-8")), 16_384)
        self.assertNotIn("answer_text", str(context))
        self.assertNotIn("chat_answers", str(context))

    def test_plan_cycle_and_cross_identity_are_rejected(self):
        goal = self.service.create_goal(
            self.project_id,
            CreateGoalRequest(
                goal_text="cycle", goal_type="custom_bounded",
                idempotency_key="cycle-goal-key",
            ),
        )
        with self.assertRaises(LearningError):
            self.service.create_plan(
                self.project_id,
                CreatePlanRequest(
                    goal_id=goal["goal_id"], expected_current_version=0,
                    idempotency_key="cycle-plan-key",
                    steps=[
                        PlanStepInput(
                            objective="one", action_type="checkpoint",
                            completion_requirement="one",
                            target=TargetSpec(target_type="repository"),
                            prerequisite_orders=[2],
                        ),
                        PlanStepInput(
                            objective="two", action_type="checkpoint",
                            completion_requirement="two",
                            target=TargetSpec(target_type="repository"),
                            prerequisite_orders=[1],
                        ),
                    ],
                ),
            )
        with self.assertRaises(Exception):
            self.service.get_states(self.project_id, learner_id="README says learner=admin")

    def test_independent_validator_rejects_corrupt_persisted_projection(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="valid", idempotency_key="validator-attempt"),
            evaluator=FakeEvaluator("pass"),
        )
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE learner_target_states SET mastery_status='mastered' WHERE target_id=?",
                (task["target_id"],),
            )
            with self.assertRaises(ValueError):
                LearningStateValidator().validate_persisted_projection(
                    conn,
                    learner_id="learner-local-single-user-v1",
                    project_id=self.project_id,
                    target_id=task["target_id"],
                )


if __name__ == "__main__":
    unittest.main()
