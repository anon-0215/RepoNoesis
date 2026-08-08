from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Mapping


QUALIFYING_TASK_TYPES = {
    "explain_symbol",
    "trace_static_relation",
    "explain_static_relationship",
    "analyze_change_impact",
    "separate_fact_inference_unknown",
}


def project_learning_events(
    events: Iterable[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = list(events)
    corrections = {
        str(outcome.get("reverses_event_id")): outcome
        for item in ordered
        if item["event_type"] == "evaluation_corrected"
        for outcome in [_json(item["validated_outcome_json"], {})]
        if outcome.get("reverses_event_id")
    }
    mastery = "unseen"
    availability = "current"
    passed_tasks: set[str] = set()
    qualifying_tasks: set[str] = set()
    last_revision = target["observed_revision"]
    last_hash = target["observed_content_hash"]
    review_reason = ""
    last_event_id = None
    previous_order = 0
    for event in ordered:
        event_order = int(event["event_order"])
        if event_order <= previous_order:
            raise ValueError("learning event order is not strictly increasing")
        previous_order = event_order
        outcome = _json(event["validated_outcome_json"], {})
        last_event_id = event["event_id"]
        if event["provenance"] == "explicit_self_report":
            if mastery == "unseen":
                mastery = "introduced"
            continue
        if (
            event["event_type"] == "attempt_evaluated"
            and event["provenance"] == "verified_assessment"
        ):
            availability = "current"
            last_revision = target["observed_revision"]
            last_hash = target["observed_content_hash"]
            correction = corrections.get(str(event["event_id"]))
            verdict = (
                correction.get("corrected_verdict")
                if correction
                else outcome.get("verdict")
            )
            task_type = (
                correction.get("task_type")
                if correction
                else outcome.get("task_type")
            )
            if verdict == "pass" and event["task_id"]:
                passed_tasks.add(str(event["task_id"]))
                if task_type in QUALIFYING_TASK_TYPES:
                    qualifying_tasks.add(str(event["task_id"]))
                mastery = (
                    "mastered"
                    if len(passed_tasks) >= 2 and qualifying_tasks
                    else "demonstrated"
                )
                review_reason = ""
            elif verdict == "partial":
                if mastery in {"unseen", "introduced", "practicing"}:
                    mastery = "practicing"
            elif verdict == "fail":
                mastery = "needs_review" if passed_tasks else "practicing"
                review_reason = "validated_failure"
        elif event["event_type"] == "revision_revalidated":
            availability = str(outcome.get("availability", "stale"))
            last_revision = event["repository_revision"]
            last_hash = str(outcome.get("content_hash", ""))
            if availability in {"changed", "missing", "ambiguous", "stale"}:
                mastery = "needs_review"
                review_reason = f"revision_{availability}"
        elif (
            event["event_type"] == "continuity_state_derived"
            and event["provenance"] == "revision_continuity"
        ):
            mapping_status = str(outcome.get("mapping_status", "unmapped"))
            availability = str(outcome.get("availability", "stale"))
            last_revision = event["repository_revision"]
            last_hash = str(outcome.get("content_hash", ""))
            if mapping_status in {"unchanged_exact", "renamed_exact"}:
                source_mastery = str(outcome.get("source_mastery_status", "unseen"))
                mastery = source_mastery if source_mastery in {
                    "unseen", "introduced", "practicing", "demonstrated", "mastered",
                    "needs_review",
                } else "needs_review"
                passed_tasks = {
                    f"continuity-pass-{index}"
                    for index in range(int(outcome.get("source_verified_pass_count", 0)))
                }
                qualifying_tasks = {
                    f"continuity-qualifying-{index}"
                    for index in range(int(outcome.get("source_qualifying_pass_count", 0)))
                }
                review_reason = ""
            else:
                mastery = "needs_review"
                review_reason = f"revision_{mapping_status}"
    return {
        "mastery_status": mastery,
        "availability": availability,
        "verified_pass_count": len(passed_tasks),
        "qualifying_pass_count": len(qualifying_tasks),
        "last_validated_revision": last_revision,
        "last_validated_content_hash": last_hash,
        "last_event_id": last_event_id,
        "review_reason": review_reason,
    }


class LearningStateValidator:
    """Re-read event history and prove that a persisted projection is authoritative."""

    COMPARED_FIELDS = (
        "mastery_status",
        "availability",
        "verified_pass_count",
        "qualifying_pass_count",
        "last_validated_revision",
        "last_validated_content_hash",
        "last_event_id",
        "review_reason",
    )

    def validate_persisted_projection(
        self,
        conn: sqlite3.Connection,
        *,
        learner_id: str,
        project_id: str,
        target_id: str,
    ) -> sqlite3.Row:
        target = conn.execute(
            """
            SELECT * FROM learning_targets
            WHERE target_id = ? AND learner_id = ? AND project_id = ?
            """,
            (target_id, learner_id, project_id),
        ).fetchone()
        state = conn.execute(
            """
            SELECT * FROM learner_target_states
            WHERE learner_id = ? AND project_id = ? AND target_id = ?
            """,
            (learner_id, project_id, target_id),
        ).fetchone()
        if not target or not state:
            raise ValueError("learning target/state ownership validation failed")
        events = conn.execute(
            """
            SELECT * FROM learning_events
            WHERE learner_id = ? AND project_id = ? AND target_id = ?
            ORDER BY event_order, event_id
            """,
            (learner_id, project_id, target_id),
        ).fetchall()
        expected = project_learning_events(events, target)
        for field in self.COMPARED_FIELDS:
            if state[field] != expected[field]:
                raise ValueError(f"learning state projection mismatch: {field}")
        if state["availability"] != "current" and state["mastery_status"] == "mastered":
            raise ValueError("stale target cannot be represented as mastered")
        return state


def _json(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback
