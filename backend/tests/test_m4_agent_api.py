from __future__ import annotations

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.database import Database
from app.services.agent_core import run_bounded_agent
from app.services.learning_contracts import (
    CreateGoalRequest,
    CreatePlanRequest,
    CreateTaskRequest,
    PlanStepInput,
    RubricCriterionInput,
    SubmitAttemptRequest,
    TargetSpec,
)
from app.services.learning_service import LearningService
from tests.m1_helpers import disabled_embedding_service
from tests.m3_helpers import make_relation_project
from tests.m4_helpers import FakeEvaluator, create_goal_plan_task
from tests.test_m2_agent import NoLlm, ScriptedPlanner, decision


class M4AgentAndApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.directory.name) / "m4-agent.sqlite")
        self.project_id, self.bundle = make_relation_project(
            self.db,
            {
                "app.py": "from helper import helper\n\ndef target():\n    return helper()\n",
                "helper.py": "def helper():\n    return 1\n",
            },
        )
        self.service = LearningService(self.db)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _run(self, planner, context):
        return run_bounded_agent(
            "解释 target",
            self.bundle,
            NoLlm(),
            self.db,
            disabled_embedding_service(),
            planner=planner,
            learning_context=context,
        )

    def test_no_goal_keeps_m3_and_adds_disabled_compatible_fields(self):
        result = self._run(
            ScriptedPlanner([
                decision("continue", "search_code", {"query": "target"}),
                decision("answer"),
            ]),
            self.service.get_learning_context(self.project_id),
        )
        self.assertEqual(result["learning_schema_version"], 1)
        self.assertEqual(result["learning_mode"], "disabled")
        self.assertTrue(result["evidence"])
        self.assertEqual(result["relation_schema_version"], 1)

    def test_learning_context_search_answer_is_bounded_and_read_only(self):
        create_goal_plan_task(self.service, self.project_id)
        before_events = self._event_count()
        result = self._run(
            ScriptedPlanner([
                decision("continue", "get_learning_context", {}),
                decision("continue", "search_code", {"query": "target"}),
                decision("answer"),
            ]),
            self.service.get_learning_context(self.project_id),
        )
        self.assertEqual([step["action"] for step in result["agent_trace"]], [
            "get_learning_context", "search_code", "answer"
        ])
        self.assertEqual(result["learning_mode"], "adaptive")
        self.assertEqual(result["learning_context_summary"]["explanation_depth"], "novice")
        self.assertIsNotNone(result["recommended_next_action"])
        self.assertEqual(self._event_count(), before_events)

    def test_learning_context_cannot_accept_identity_or_be_called_twice(self):
        create_goal_plan_task(self.service, self.project_id)
        result = self._run(
            ScriptedPlanner([
                decision("continue", "get_learning_context", {"learner_id": "attacker"}),
                decision("continue", "get_learning_context", {}),
                decision("continue", "get_learning_context", {}),
                decision("answer"),
            ]),
            self.service.get_learning_context(self.project_id),
        )
        calls = [step["tool_calls"][0] for step in result["agent_trace"] if step["tool_calls"]]
        # The invalid identity-bearing input is rejected by the Planner contract
        # before execution; the repair call succeeds, and the next duplicate is
        # still rejected by the existing per-run tool limit.
        self.assertEqual(calls[0]["status"], "succeeded")
        self.assertEqual(calls[-1]["status"], "rejected")

    def test_agent_trace_never_leaks_user_attempt_answer(self):
        _goal, _plan, task = create_goal_plan_task(self.service, self.project_id)
        secret_answer = "private-learning-answer-should-not-leak"
        self.service.submit_attempt(
            self.project_id,
            task["task_id"],
            SubmitAttemptRequest(answer_text=secret_answer, idempotency_key="private-attempt-key"),
            evaluator=FakeEvaluator("pass"),
        )
        result = self._run(
            ScriptedPlanner([
                decision("continue", "get_learning_context", {}),
                decision("continue", "search_code", {"query": "target"}),
                decision("answer"),
            ]),
            self.service.get_learning_context(self.project_id),
        )
        self.assertNotIn(secret_answer, str(result["agent_trace"]))
        self.assertNotIn(secret_answer, str(result["learning_context_summary"]))

    def test_api_routes_bind_server_learner_and_old_ask_shape(self):
        main = importlib.import_module("app.main")
        with (
            patch.object(main, "db", self.db),
            patch.object(main, "learning_service", self.service),
            patch.object(main, "llm", NoLlm()),
            patch.object(main, "embedding_service", disabled_embedding_service()),
        ):
            goal = main.create_learning_goal(
                self.project_id,
                CreateGoalRequest(
                    goal_text="理解项目",
                    goal_type="repository_onboarding",
                    idempotency_key="api-goal-key",
                ),
            )
            goals = main.get_learning_goals(self.project_id)
            result = main.ask_project(
                self.project_id, main.AskRequest(question="target")
            )
            validated = main.AskResponse.model_validate(result)
        self.assertEqual(goal["goal_id"], goals["items"][0]["goal_id"])
        self.assertEqual(validated.learning_schema_version, 1)
        self.assertIn(validated.learning_mode, {"profiled", "adaptive"})
        self.assertEqual(validated.evidence_schema_version, 1)
        self.assertEqual(validated.agent_schema_version, 1)
        self.assertEqual(validated.relation_schema_version, 1)

    def test_two_different_passed_tasks_are_required_for_mastered(self):
        _goal, _plan, first_task = create_goal_plan_task(self.service, self.project_id)
        first = self.service.submit_attempt(
            self.project_id,
            first_task["task_id"],
            SubmitAttemptRequest(answer_text="first", idempotency_key="mastery-attempt-one"),
            evaluator=FakeEvaluator("pass"),
        )
        second_task = self._task_for_current_step("task-key-master-two")
        second = self.service.submit_attempt(
            self.project_id,
            second_task["task_id"],
            SubmitAttemptRequest(answer_text="second", idempotency_key="mastery-attempt-two"),
            evaluator=FakeEvaluator("pass"),
        )
        self.assertEqual(first["learner_state"]["mastery_status"], "demonstrated")
        self.assertEqual(second["learner_state"]["mastery_status"], "mastered")
        self.assertEqual(second["learner_state"]["verified_pass_count"], 2)

    def test_failure_after_verified_pass_requires_review_without_deleting_success(self):
        _goal, _plan, first_task = create_goal_plan_task(self.service, self.project_id)
        self.service.submit_attempt(
            self.project_id,
            first_task["task_id"],
            SubmitAttemptRequest(answer_text="first", idempotency_key="review-attempt-one"),
            evaluator=FakeEvaluator("pass"),
        )
        second_task = self._task_for_current_step("task-key-review-two")
        failed = self.service.submit_attempt(
            self.project_id,
            second_task["task_id"],
            SubmitAttemptRequest(answer_text="wrong", idempotency_key="review-attempt-two"),
            evaluator=FakeEvaluator("fail"),
        )
        self.assertEqual(failed["learner_state"]["mastery_status"], "needs_review")
        self.assertEqual(failed["learner_state"]["verified_pass_count"], 1)
        self.assertEqual(failed["learning_plan"]["adaptation_reason"], "validated_failure_requires_remediation")

    def _task_for_current_step(self, key: str) -> dict:
        plan = self.service.get_current_plan(self.project_id)
        step = next(item for item in plan["steps"] if item["status"] == "active")
        return self.service.create_task(
            self.project_id,
            CreateTaskRequest(
                plan_id=plan["plan_id"],
                plan_version=plan["version"],
                step_id=step["step_id"],
                task_type="explain_symbol",
                prompt_text="再次解释 target",
                rubric=[RubricCriterionInput(
                    criterion_id="source_fact",
                    criterion_type="source_fact",
                    weight=1.0,
                    expected_claim="说明 target 的静态职责",
                    critical=True,
                )],
                idempotency_key=key,
            ),
        )

    def _event_count(self) -> int:
        with self.db.connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]


if __name__ == "__main__":
    unittest.main()
