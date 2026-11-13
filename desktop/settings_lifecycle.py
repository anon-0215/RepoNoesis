"""Desktop-only restart lease; counts actual ask workers, even after stream disconnect."""
from functools import wraps
import threading
import time


class SettingsRestartGate:
    def __init__(self, clock=time.monotonic, busy=None):
        self.lock = threading.Lock()
        self.active = 0
        self.lease = None
        self.expires = 0
        self.clock = clock
        self.busy = busy

    def _expire(self):
        if self.lease is not None and self.clock() >= self.expires:
            self.lease = None

    def acquire(self, lease):
        with self.lock:
            self._expire()
            if self.active or (self.busy is not None and self.busy()) or (self.lease is not None and self.lease != lease):
                return False
            self.lease, self.expires = lease, self.clock() + 30
            return True

    def release(self, lease):
        with self.lock:
            if self.lease == lease:
                self.lease = None

    def wrap(self, function):
        @wraps(function)
        def run(*args, **kwargs):
            from fastapi import HTTPException
            with self.lock:
                self._expire()
                if self.lease is not None:
                    raise HTTPException(503, "本地服务正在应用设置，请就绪后重试。")
                self.active += 1
            try:
                return function(*args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1
        return run


def install(app, main, instance):
    from fastapi import Header, HTTPException
    from pydantic import BaseModel, Field
    from typing import Literal
    def pending_import():
        if not hasattr(main, 'db'):
            return False  # Lightweight offline fixtures have no repository store.
        try:
            with main.db.connect() as connection:
                return connection.execute("SELECT 1 FROM repository_update_runs WHERE status IN ('pending','running') LIMIT 1").fetchone() is not None
        except Exception:
            return True  # An unreadable store cannot establish an idle state.
    gate = SettingsRestartGate(busy=pending_import)
    main._execute_ask = gate.wrap(main._execute_ask)
    # FastAPI retains the registered callable. Count the actual synchronous
    # import worker from before cloning until it exits, even after disconnect.
    if hasattr(main, 'analyze_project'):
        original = main.analyze_project
        main.analyze_project = gate.wrap(original)
        for route in app.routes:
            dependant = getattr(route, 'dependant', None)
            if dependant is not None and dependant.call is original:
                dependant.call = main.analyze_project
    if hasattr(main, '_update_service'):
        factory = main._update_service
        def guarded_update_service():
            service = factory()
            for name in ('start_refresh', 'retry_run', 'execute_run'):
                setattr(service, name, gate.wrap(getattr(service, name)))
            return service
        main._update_service = guarded_update_service

    class RestartRequest(BaseModel):
        action: Literal["acquire", "release"]
        lease_id: str = Field(pattern=r"^[0-9a-f]{32}$")

    @app.post("/api/desktop/settings-restart")
    def restart(request: RestartRequest, x_reponoesis_instance: str = Header(default="")):
        if not instance or x_reponoesis_instance != instance:
            raise HTTPException(409, "桌面服务身份已变化，请重新打开设置。")
        if request.action == "release":
            gate.release(request.lease_id)
        elif not gate.acquire(request.lease_id):
            raise HTTPException(409, "正在问答、导入项目或应用设置，请结束后再保存。")
        return {"ok": True, "instance": instance}

    return gate
