"""Desktop-only restart lease; counts actual ask workers, even after stream disconnect."""
from functools import wraps
import threading
import time


class SettingsRestartGate:
    def __init__(self, clock=time.monotonic):
        self.lock = threading.Lock()
        self.active = 0
        self.lease = None
        self.expires = 0
        self.clock = clock

    def _expire(self):
        if self.lease is not None and self.clock() >= self.expires:
            self.lease = None

    def acquire(self, lease):
        with self.lock:
            self._expire()
            if self.active or (self.lease is not None and self.lease != lease):
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
    gate = SettingsRestartGate()
    main._execute_ask = gate.wrap(main._execute_ask)

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
            raise HTTPException(409, "正在生成回答或应用设置，请结束后再保存。")
        return {"ok": True, "instance": instance}

    return gate
