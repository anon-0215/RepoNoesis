"""Content-free progress projection from request-owned diagnostics calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.services.smoke_diagnostics import SmokeDiagnosticsRecorder


class ProgressRecorder(SmokeDiagnosticsRecorder):
    def __init__(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        super().__init__()
        self._emit = emit
        self._answer_started = False
        self._relation_applicable = False
        self._evidence_count: int | None = None
        self._validation_checkpoint: str | None = None

    def set_validation_checkpoint(self, checkpoint: str) -> None:
        self._validation_checkpoint = checkpoint

    def record_evidence_count(self, count: int) -> None:
        super().record_evidence_count(count)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            self._evidence_count = count

    def set_relation_applicable(self, applicable: bool) -> None:
        self._relation_applicable = applicable

    def begin_base(self) -> None:
        self._emit("base_started", {})

    def enter_stage(self, stage: str) -> None:
        previous = self.stage
        super().enter_stage(stage)
        if stage == "agent_planner" and previous != stage:
            self._emit("planner_started", {})
        elif stage == "final_answer" and previous != stage:
            self._answer_started = True
            self._emit("answer_started", {})

    def record_base_retrieval(self, **kwargs: Any) -> None:
        super().record_base_retrieval(**kwargs)
        self._emit("base_completed", {
            "status": kwargs["status"],
            "new_evidence_count": kwargs["new_evidence_count"],
        })

    def record_tool_execution(self, **kwargs: Any) -> None:
        super().record_tool_execution(**kwargs)
        if kwargs["phase"] == "planner":
            self._emit("tool_completed", {
                "tool": kwargs["tool_name"],
                "status": kwargs["status"],
                "evidence_added": kwargs["evidence_added"],
            })

    def record_planner_enhancement_termination(self, reason: str) -> None:
        super().record_planner_enhancement_termination(reason)
        self._emit("planner_stopped", {"reason": reason})

    def mark_citation_validation_completed(self, *, passed: bool) -> None:
        super().mark_citation_validation_completed(passed=passed)
        # Empty Evidence validates vacuously in the existing validator. Keep
        # its diagnostic record, but do not announce a user-facing pass.
        if self._evidence_count == 0:
            return
        self._emit("citation_checked", {
            "passed": passed,
            "phase": "after_answer" if self._validation_checkpoint == "post_answer_evidence" or (self._validation_checkpoint is None and self._answer_started) else "before_answer",
            "checkpoint": self._validation_checkpoint,
        })

    def mark_relation_validation_completed(self, *, passed: bool) -> None:
        super().mark_relation_validation_completed(passed=passed)
        if not self._relation_applicable:
            return
        self._emit("relation_checked", {
            "passed": passed,
            "phase": "after_answer" if self._validation_checkpoint == "post_answer_evidence" or (self._validation_checkpoint is None and self._answer_started) else "before_answer",
            "checkpoint": self._validation_checkpoint,
        })
