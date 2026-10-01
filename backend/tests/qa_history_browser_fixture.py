"""Explicit opt-in browser fixture: temporary workspace DB + deterministic answer adapter.

Run from backend with --database under output/playwright/rnfe03/ and an unused --port.
Never uses app.run_server, .env loading, formal data, remote models or repository imports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frontend-port", type=int, required=True)
    args = parser.parse_args()
    allowed = Path(__file__).resolve().parents[2] / "output" / "playwright" / "rnfe03"
    target = Path(args.database).resolve()
    if allowed.resolve() not in target.parents or target.exists():
        raise ValueError("Fixture requires a new database inside its isolated output directory.")
    os.environ.update(GITLEARN_DB=str(target), LLM_API_KEY="", EMBEDDING_ENABLED="false",
                      RUNTIME_DATA_DIR=str(allowed / "runtime"))
    from app import main as routes
    from app.database import Database
    from fastapi.middleware.cors import CORSMiddleware
    from tests.m1_helpers import make_project, make_chunk, disabled_embedding_service
    from tests.test_chat_history import add_snapshot
    from tests.test_ask_execution_modes import _RouteLlm
    import uvicorn

    db = Database(target)
    content = "def authenticate_user(password):\n    return verify(password)\n"
    old, bundle = make_project(db, [("src/auth.py", "authenticate_user", content)])
    workspace = db.get_workspace_for_project(old)["id"]
    citation = {"evidence_id": "E5", "path": "src/auth.py", "summary": "authenticate_user",
                "qualified_name": "authenticate_user", "snippet": content, "start_line": 1, "end_line": 2}
    for i in range(8):
        db.save_chat_answer(old, f"旧版本问题 {i + 1}", "旧版本保存答案 [E5]", [citation])
    revision = "b" * 40
    current = add_snapshot(db, workspace, old, revision)
    files = [{"path": "src/auth.py", "content": content, "is_core": True, "summary": "authenticate_user"}]
    db.save_analysis(current, {"files": files, "modules": [], "overview": "隔离问答历史验收项目"}, files, [],
                     [make_chunk("src/auth.py", "authenticate_user", content, revision=revision)])
    for i in range(15):
        db.save_chat_answer(current, f"旧格式问题 {i + 1}", "旧格式保存答案 [E5]", [citation])
    other, _ = make_project(db, [("src/auth.py", "authenticate_user", content)])
    with db.connect() as conn:
        conn.execute("UPDATE repository_workspaces SET display_name='历史恢复验收' WHERE id=?", (workspace,))
        conn.execute("UPDATE repository_workspaces SET display_name='另一个隔离项目' WHERE active_project_id=?", (other,))

    class Deterministic(_RouteLlm):
        def chat(self, messages, **kwargs):
            if kwargs.get("purpose") == "planner":
                return super().chat(messages, **kwargs)
            # A visible pending phase without any remote Provider call.
            time.sleep(0.8)
            return json.dumps({"parts": [{"text": "隔离验证答案：authenticate_user 使用已保存源码中的校验函数。", "evidence_aliases": ["A1"]}]}, ensure_ascii=False)
    class Learning:
        def get_learning_context(self, _project): return None
    routes.db = db
    routes.llm = Deterministic()
    routes.embedding_service = disabled_embedding_service()
    routes.learning_service = Learning()
    routes.app.add_middleware(CORSMiddleware, allow_origins=[f"http://127.0.0.1:{args.frontend_port}"],
                             allow_methods=["GET", "POST"], allow_headers=["*"])
    saved = routes.ask_project(current, routes.AskRequest(question="authenticate_user", execution_mode="rag", repository_revision=revision))
    allowed.mkdir(parents=True, exist_ok=True)
    (allowed / "fixture-identity.json").write_text(json.dumps({
        "workspace_id": workspace, "project_id": current, "revision": revision,
        "old_project_id": old, "other_workspace_id": db.get_workspace_for_project(other)["id"],
        "seed_record_id": saved["record_id"],
    }), encoding="utf-8")
    uvicorn.run(routes.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
