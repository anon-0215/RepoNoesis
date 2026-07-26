from __future__ import annotations


LEARNING_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS learner_profiles (
        learner_id TEXT PRIMARY KEY,
        profile_type TEXT NOT NULL CHECK(profile_type = 'local_single_user'),
        status TEXT NOT NULL CHECK(status IN ('active', 'disabled')),
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_goals (
        goal_id TEXT PRIMARY KEY,
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        created_revision TEXT NOT NULL,
        goal_text TEXT NOT NULL CHECK(length(goal_text) BETWEEN 1 AND 2000),
        goal_type TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('active', 'completed', 'cancelled')),
        idempotency_key TEXT NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(learner_id, project_id, idempotency_key),
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_targets (
        target_id TEXT PRIMARY KEY,
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        observed_revision TEXT NOT NULL,
        target_type TEXT NOT NULL,
        normalized_path TEXT NOT NULL DEFAULT '',
        qualified_name TEXT NOT NULL DEFAULT '',
        bounded_concept TEXT NOT NULL DEFAULT '',
        code_chunk_id INTEGER,
        observed_content_hash TEXT NOT NULL DEFAULT '',
        availability TEXT NOT NULL DEFAULT 'current',
        resolution_status TEXT NOT NULL DEFAULT 'resolved',
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(learner_id, project_id, observed_revision, target_type, normalized_path, qualified_name, bounded_concept),
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY(code_chunk_id) REFERENCES code_chunks(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_plans (
        plan_id TEXT PRIMARY KEY,
        goal_id TEXT NOT NULL,
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        source_revision TEXT NOT NULL,
        plan_version INTEGER NOT NULL CHECK(plan_version >= 1),
        status TEXT NOT NULL CHECK(status IN ('active', 'completed', 'superseded', 'stale')),
        adapted INTEGER NOT NULL DEFAULT 0,
        adaptation_reason TEXT NOT NULL DEFAULT '',
        idempotency_key TEXT NOT NULL,
        superseded_by TEXT,
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(goal_id, plan_version),
        UNIQUE(learner_id, project_id, idempotency_key),
        FOREIGN KEY(goal_id) REFERENCES learning_goals(goal_id) ON DELETE CASCADE,
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY(superseded_by) REFERENCES learning_plans(plan_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_plan_steps (
        step_id TEXT PRIMARY KEY,
        plan_id TEXT NOT NULL,
        step_order INTEGER NOT NULL CHECK(step_order >= 1),
        target_id TEXT NOT NULL,
        objective TEXT NOT NULL CHECK(length(objective) BETWEEN 1 AND 1000),
        action_type TEXT NOT NULL,
        completion_requirement TEXT NOT NULL CHECK(length(completion_requirement) BETWEEN 1 AND 1000),
        status TEXT NOT NULL CHECK(status IN ('pending', 'active', 'completed', 'needs_review', 'skipped', 'invalid')),
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(plan_id, step_order),
        FOREIGN KEY(plan_id) REFERENCES learning_plans(plan_id) ON DELETE CASCADE,
        FOREIGN KEY(target_id) REFERENCES learning_targets(target_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_step_prerequisites (
        plan_id TEXT NOT NULL,
        step_id TEXT NOT NULL,
        prerequisite_step_id TEXT NOT NULL,
        PRIMARY KEY(plan_id, step_id, prerequisite_step_id),
        FOREIGN KEY(plan_id) REFERENCES learning_plans(plan_id) ON DELETE CASCADE,
        FOREIGN KEY(step_id) REFERENCES learning_plan_steps(step_id) ON DELETE CASCADE,
        FOREIGN KEY(prerequisite_step_id) REFERENCES learning_plan_steps(step_id) ON DELETE CASCADE,
        CHECK(step_id != prerequisite_step_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_tasks (
        task_id TEXT PRIMARY KEY,
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        repository_revision TEXT NOT NULL,
        goal_id TEXT NOT NULL,
        plan_id TEXT NOT NULL,
        plan_version INTEGER NOT NULL,
        step_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        task_type TEXT NOT NULL,
        prompt_text TEXT NOT NULL CHECK(length(prompt_text) BETWEEN 1 AND 2000),
        rubric_version INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL CHECK(status IN ('active', 'completed', 'stale', 'invalid')),
        idempotency_key TEXT NOT NULL,
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(learner_id, project_id, idempotency_key),
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY(goal_id) REFERENCES learning_goals(goal_id) ON DELETE CASCADE,
        FOREIGN KEY(plan_id) REFERENCES learning_plans(plan_id),
        FOREIGN KEY(step_id) REFERENCES learning_plan_steps(step_id),
        FOREIGN KEY(target_id) REFERENCES learning_targets(target_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_task_evidence (
        task_id TEXT NOT NULL,
        evidence_id TEXT NOT NULL,
        code_chunk_id INTEGER,
        repository_revision TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        PRIMARY KEY(task_id, evidence_id),
        FOREIGN KEY(task_id) REFERENCES learning_tasks(task_id) ON DELETE CASCADE,
        FOREIGN KEY(code_chunk_id) REFERENCES code_chunks(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_rubric_criteria (
        task_id TEXT NOT NULL,
        criterion_id TEXT NOT NULL,
        criterion_type TEXT NOT NULL,
        weight REAL NOT NULL CHECK(weight > 0 AND weight <= 1),
        expected_claim TEXT NOT NULL CHECK(length(expected_claim) BETWEEN 1 AND 1000),
        critical INTEGER NOT NULL DEFAULT 0,
        supporting_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        schema_version INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY(task_id, criterion_id),
        FOREIGN KEY(task_id) REFERENCES learning_tasks(task_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_attempts (
        attempt_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_revision TEXT NOT NULL,
        answer_text TEXT NOT NULL CHECK(length(answer_text) BETWEEN 1 AND 12000),
        idempotency_key TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('submitted', 'evaluated', 'ungradable')),
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(task_id, idempotency_key),
        FOREIGN KEY(task_id) REFERENCES learning_tasks(task_id),
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_evaluations (
        evaluation_id TEXT PRIMARY KEY,
        attempt_id TEXT NOT NULL UNIQUE,
        evaluator_schema_version INTEGER NOT NULL DEFAULT 1,
        verdict TEXT NOT NULL CHECK(verdict IN ('pass', 'partial', 'fail', 'ungradable')),
        criterion_results_json TEXT NOT NULL DEFAULT '[]',
        supported_feedback_json TEXT NOT NULL DEFAULT '[]',
        missing_concepts_json TEXT NOT NULL DEFAULT '[]',
        misconceptions_json TEXT NOT NULL DEFAULT '[]',
        used_evidence_ids_json TEXT NOT NULL DEFAULT '[]',
        warnings_json TEXT NOT NULL DEFAULT '[]',
        validated INTEGER NOT NULL DEFAULT 1,
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(attempt_id) REFERENCES learning_attempts(attempt_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learning_events (
        event_id TEXT PRIMARY KEY,
        idempotency_key TEXT NOT NULL UNIQUE,
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        repository_revision TEXT NOT NULL,
        goal_id TEXT,
        plan_id TEXT,
        step_id TEXT,
        target_id TEXT NOT NULL,
        task_id TEXT,
        attempt_id TEXT,
        evaluation_id TEXT,
        event_type TEXT NOT NULL,
        provenance TEXT NOT NULL,
        validated_outcome_json TEXT NOT NULL DEFAULT '{}',
        event_order INTEGER NOT NULL,
        state_update_rule_version INTEGER NOT NULL DEFAULT 1,
        schema_version INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY(goal_id) REFERENCES learning_goals(goal_id),
        FOREIGN KEY(plan_id) REFERENCES learning_plans(plan_id),
        FOREIGN KEY(step_id) REFERENCES learning_plan_steps(step_id),
        FOREIGN KEY(target_id) REFERENCES learning_targets(target_id),
        FOREIGN KEY(task_id) REFERENCES learning_tasks(task_id),
        FOREIGN KEY(attempt_id) REFERENCES learning_attempts(attempt_id),
        FOREIGN KEY(evaluation_id) REFERENCES learning_evaluations(evaluation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS learner_target_states (
        learner_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        mastery_status TEXT NOT NULL,
        availability TEXT NOT NULL,
        verified_pass_count INTEGER NOT NULL DEFAULT 0,
        qualifying_pass_count INTEGER NOT NULL DEFAULT 0,
        last_validated_revision TEXT NOT NULL DEFAULT '',
        last_validated_content_hash TEXT NOT NULL DEFAULT '',
        last_event_id TEXT,
        review_reason TEXT NOT NULL DEFAULT '',
        state_update_rule_version INTEGER NOT NULL DEFAULT 1,
        schema_version INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(learner_id, project_id, target_id),
        FOREIGN KEY(learner_id) REFERENCES learner_profiles(learner_id),
        FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
        FOREIGN KEY(target_id) REFERENCES learning_targets(target_id),
        FOREIGN KEY(last_event_id) REFERENCES learning_events(event_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_learning_goals_active ON learning_goals (learner_id, project_id, status, updated_at)",
    "CREATE INDEX IF NOT EXISTS idx_learning_plans_version ON learning_plans (goal_id, plan_version DESC)",
    "CREATE INDEX IF NOT EXISTS idx_learning_steps_order ON learning_plan_steps (plan_id, step_order)",
    "CREATE INDEX IF NOT EXISTS idx_learning_targets_owner ON learning_targets (learner_id, project_id, target_type, normalized_path, qualified_name)",
    "CREATE INDEX IF NOT EXISTS idx_learning_targets_revision ON learning_targets (project_id, observed_revision, availability)",
    "CREATE INDEX IF NOT EXISTS idx_learning_attempts_task ON learning_attempts (task_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_learning_events_owner_time ON learning_events (learner_id, project_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_learning_states_review ON learner_target_states (learner_id, project_id, mastery_status, availability)",
    """
    CREATE TRIGGER IF NOT EXISTS trg_learning_events_no_update
    BEFORE UPDATE ON learning_events
    BEGIN SELECT RAISE(ABORT, 'learning events are immutable'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_learning_events_no_delete
    BEFORE DELETE ON learning_events
    BEGIN SELECT RAISE(ABORT, 'learning events are immutable'); END
    """,
)
