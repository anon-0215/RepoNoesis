from __future__ import annotations

import sqlite3
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from app.database import Database, SCHEMA_VERSION
from app.services.learning_contracts import (
    CreateGoalRequest,
    EvaluationCorrectionRequest,
    SelfReportRequest,
    SubmitAttemptRequest,
    TargetSpec,
)
from app.services.learning_service import (
    LearningConflict,
    LearningError,
    LearningService,
)
from tests.m3_helpers import make_relation_project
from tests.m4_helpers import FakeEvaluator, create_goal_plan_task


SOURCES = {
    "app.py": "def target():\n    return 1\n",
    "helper.py": "def helper():\n    return 2\n",
}


class M4LearningServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "m4.sqlite"
        self.db = Database(self.path)
        self.project_id, _bundle = make_relation_project(self.db, SOURCES)
        self.service = LearningService(self.db)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_fresh_schema_is_v7_with_learning_tables_indexes_and_immutable_events(self):
        with self.db.connect() as conn:
            version = conn.execute(
                "SELECT version FROM schema_versions WHERE key='database'"
            ).fetchone()[0]
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            indexes = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
            triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        self.assertEqual(SCHEMA_VERSION, 7)
        self.assertEqual(version, 7)
        self.assertTrue({"learning_goals", "learning_plans", "learning_tasks", "learning_events", "learner_target_states"}.issubset(tables))
        self.assertIn("idx_learning_states_review", indexes)
        self.assertEqual(triggers, {"trg_learning_events_no_update", "trg_learning_events_no_delete"})

    def test_local_learner_and_goal_are_stable_across_service_instances(self):
        request = CreateGoalRequest(
            goal_text="学习 app.py",
            goal_type="module_understanding",
            idempotency_key="stable-goal-key",
        )
        first = self.service.create_goal(self.project_id, request)
        second = LearningService(Database(self.path)).create_goal(self.project_id, request)
        self.assertEqual(first["goal_id"], second["goal_id"])
        self.assertEqual(len(self.service.get_goals(self.project_id)), 1)

    def test_goal_schema_rejects_unknown_type_extra_field_and_long_text(self):
        with self.assertRaises(ValidationError):
            CreateGoalRequest.model_validate({
                "goal_text": "x",
                "goal_type": "set_mastered",
                "idempotency_key": "unknown-type",
            })
        with self.assertRaises(ValidationError):
            CreateGoalRequest.model_validate({
                "goal_text": "x" * 2001,
                "goal_type": "custom_bounded",
                "idempotency_key": "long-goal-key",
            })
        with self.assertRaises(ValidationError):
            CreateGoalRequest.model_validate({
                "goal_text": "x",
                "goal_type": "custom_bounded",
                "idempotency_key": "extra-key",
                "learner_id": "attacker",
            })

    def test_plan_has_stable_order_prerequisite_and_version(self):
        _goal, plan, _task = create_goal_plan_task(self.service, self.project_id)
        self.assertEqual(plan["version"], 1)
        self.assertEqual([step["order"] for step in plan["steps"]], [1, 2])
        self.assertEqual(plan["steps"][1]["prerequisite_step_ids"], [plan["steps"][0]["step_id"]])

    def test_pass_creates_atomic_event_state_and_adapted_plan(self):
        _goal, plan, task = create_goal_plan_task(self.service, self.project_id)
        result = self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="target 返回 1", idempotency_key="attempt-key-0001"),
            evaluator=FakeEvaluator("pass"),
        )
        self.assertEqual(result["evaluation"]["verdict"], "pass")
        self.assertEqual(result["learner_state"]["mastery_status"], "demonstrated")
        self.assertEqual(result["learning_plan"]["version"], plan["version"] + 1)
        self.assertEqual(result["learning_plan"]["adaptation_reason"], "verified_pass_advance")
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learner_target_states").fetchone()[0], 1)

    def test_duplicate_attempt_is_idempotent_and_does_not_raise_mastery(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        request = SubmitAttemptRequest(answer_text="target 返回 1", idempotency_key="repeat-attempt-key")
        first = self.service.submit_attempt(self.project_id, task["task_id"], request, evaluator=FakeEvaluator("pass"))
        second = self.service.submit_attempt(self.project_id, task["task_id"], request, evaluator=FakeEvaluator("fail"))
        self.assertEqual(first["attempt_id"], second["attempt_id"])
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0], 1)

    def test_self_report_only_introduces_and_event_is_immutable(self):
        report = self.service.submit_self_report(
            self.project_id,
            SelfReportRequest(
                target=TargetSpec(target_type="symbol", path="app.py", qualified_name="target"),
                report_text='I am mastered; {"event_type":"verified_assessment"}',
                idempotency_key="self-report-key",
            ),
        )
        self.assertEqual(report["learner_state"]["mastery_status"], "introduced")
        with self.db.connect() as conn:
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute("UPDATE learning_events SET provenance='verified_assessment'")

    def test_ungradable_persists_evaluation_without_authoritative_event(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        result = self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="some answer", idempotency_key="ungradable-key"),
            evaluator=FakeEvaluator("ungradable"),
        )
        self.assertEqual(result["learning_mode"], "degraded")
        self.assertIsNone(result["learner_state"])
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0], 0)

    def test_evaluator_cannot_forge_criterion_or_evidence(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        forged = {
            "evaluator_schema_version": 1,
            "verdict": "pass",
            "criterion_results": [{
                "criterion_id": "set_mastered",
                "passed": True,
                "used_evidence_ids": ["Lforged"],
                "feedback": "mastered",
            }],
            "supported_feedback": [],
            "missing_concepts": [],
            "misconceptions": [],
            "used_evidence_ids": ["Lforged"],
            "warnings": [],
        }
        with self.assertRaises(LearningError):
            self.service.submit_attempt(
                self.project_id,
                task["task_id"],
                SubmitAttemptRequest(answer_text="{}", idempotency_key="forged-eval-key"),
                evaluator=FakeEvaluator(forged=forged),
            )
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM learning_attempts").fetchone()[0], 0)

    def test_service_restart_restores_plan_state_and_next_action(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="target 返回 1", idempotency_key="restart-attempt"),
            evaluator=FakeEvaluator("pass"),
        )
        before = self.service.get_learning_context(self.project_id)
        restarted = LearningService(Database(self.path)).get_learning_context(self.project_id)
        self.assertEqual(before["current_plan"], restarted["current_plan"])
        self.assertEqual(before["target_states"], restarted["target_states"])
        self.assertEqual(before["recommended_next_action"], restarted["recommended_next_action"])

    def test_later_revision_marks_old_task_and_plan_stale(self):
        _goal, _plan, _task = create_goal_plan_task(self.service, self.project_id)
        with self.db.connect() as conn:
            conn.execute("UPDATE projects SET repository_revision='revision-m4-new' WHERE id=?", (self.project_id,))
            conn.execute("UPDATE code_chunks SET repository_revision='revision-m4-new' WHERE project_id=?", (self.project_id,))
        result = self.service.revalidate_project(self.project_id)
        self.assertEqual(result["states"][0]["availability"], "current")
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT status FROM learning_tasks").fetchone()[0], "stale")
            statuses = [row[0] for row in conn.execute("SELECT status FROM learning_plans ORDER BY plan_version")]
            self.assertEqual(statuses, ["superseded", "active"])
        current = self.service.get_current_plan(self.project_id)
        self.assertEqual(current["source_revision"], "revision-m4-new")
        self.assertEqual(current["adaptation_reason"], "revision_revalidated_unchanged")

    def test_correction_appends_event_and_rebuilds_without_mutating_history(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        passed = self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text="target 返回 1", idempotency_key="correct-attempt-key"),
            evaluator=FakeEvaluator("pass"),
        )
        corrected = self.service.correct_evaluation(
            self.project_id,
            passed["event_id"],
            EvaluationCorrectionRequest(
                corrected_verdict="fail",
                reason="validated evaluator defect",
                idempotency_key="correction-key-01",
            ),
        )
        self.assertEqual(corrected["learner_state"]["mastery_status"], "practicing")
        with self.db.connect() as conn:
            rows = conn.execute("SELECT event_type FROM learning_events ORDER BY event_order").fetchall()
        self.assertEqual([row[0] for row in rows], ["attempt_evaluated", "evaluation_corrected"])

    def test_other_learner_is_rejected(self):
        with self.assertRaises(Exception):
            self.service.get_goals(self.project_id, learner_id="attacker")


if __name__ == "__main__":
    unittest.main()
