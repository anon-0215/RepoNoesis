"""Bounded local isolation for a single blocking urllib attempt."""
from pathlib import Path
from contextvars import ContextVar
import hashlib
import json
import subprocess
import sys
import threading
import time
import uuid

_AUDIT = ContextVar("provider_transport_audit", default=None)


class TransportCollectionError(RuntimeError):
    """Opt-in collection failed; the experiment must not report success."""


class transport_audit_scope:
    """Opt-in safe receipts and worker policy, separate from product defaults."""
    def __init__(self, callback, *, policy):
        self.value = (callback, policy)

    def __enter__(self):
        self.token = _AUDIT.set(self.value)
        return self

    def __exit__(self, *_args):
        _AUDIT.reset(self.token)
        return False


def prepared_parameters(request):
    payload = json.loads(request.data)
    values = {key: payload[key] for key in (
        "model", "thinking", "reasoning_effort", "temperature", "max_tokens"
    ) if key in payload}
    values["messages_sha256"] = hashlib.sha256(json.dumps(
        payload.get("messages"), ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return values


# A stuck OS teardown retains its slot until it really exits. There can never
# be an unbounded succession of detached workers, pipes, or reaper threads.
_SLOTS = threading.BoundedSemaphore(4)
_WORKERS = set()
_LOCK = threading.Lock()


def active_worker_count():
    with _LOCK:
        return len(_WORKERS)


def _send(stream, envelope):
    try:
        stream.write(envelope)
        stream.flush()
    except (OSError, ValueError):
        pass  # Kill/exit closes the peer; the owner alone decides the outcome.
    finally:
        try:
            stream.close()
        except (OSError, ValueError):
            pass


def _release(process, writer, input_stream):
    try:
        process.communicate()
        if writer is not None:
            writer.join()
    finally:
        for stream in (input_stream, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        with _LOCK:
            _WORKERS.discard(process)
        _SLOTS.release()


def isolated_response(request, *, timeout, check_active, on_headers=None, audit=None):
    """Return private protocol records; watchdog includes worker startup/parse.

    ``check_active`` enforces the owner's absolute deadline and cancellation.
    Single-attempt timeout is also absolute, not a per-socket-operation reset.
    No worker can mutate a request, recorder, Store, or history.
    """
    attempt_end = time.monotonic() + timeout
    audit = audit if audit is not None else {}
    audit.update(network_state="not_started", local_reason="slot_wait")
    audit.setdefault("attempt_id", uuid.uuid4().hex)
    observer = _AUDIT.get()

    def observe(record):
        if observer:
            try:
                observer[0]({**record, **{key: audit[key] for key in (
                    "attempt_id", "logical_call_id", "attempt_index", "request_id", "provider_call_id"
                ) if key in audit}})
            except Exception as exc:
                raise TransportCollectionError("Transport collection failed") from exc

    def check():
        check_active()
        if time.monotonic() >= attempt_end:
            raise TimeoutError()

    try:
        if observer:
            observe({"event": "parent_prepared", "parameters": prepared_parameters(request)})
        while True:
            check()
            if _SLOTS.acquire(timeout=min(0.02, max(0, attempt_end - time.monotonic()))):
                break
    except BaseException:
        observe({"event": "transport_finished", **audit})
        raise
    process = None
    writer = None
    input_stream = None
    output = b""
    try:
        check()
        audit["local_reason"] = "worker_start_failed"
        worker = Path(__file__).with_name("provider_transport_worker.py")
        process = subprocess.Popen([sys.executable, "-I", "-B", str(worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with _LOCK:
            _WORKERS.add(process)
        audit.update(network_state="unknown", local_reason="worker_state_unconfirmed")
        envelope = json.dumps({"url": request.full_url,
            "body": request.data.decode("utf-8"), "headers": dict(request.header_items()),
            "timeout": max(0.001, attempt_end - time.monotonic()),
            "attempt_id": audit["attempt_id"],
            "collect_parameters": observer is not None,
            "audit_policy": observer[1] if observer else None}).encode()
        input_stream = process.stdin
        # Windows communicate(input=...) writes stdin synchronously before its
        # timeout takes effect. Keep that potentially blocking write off the
        # owner thread, under the same retained process slot.
        process.stdin = None
        writer = threading.Thread(target=_send, args=(input_stream, envelope), daemon=True)
        writer.start()
        while True:
            check()
            try:
                output, _ = process.communicate(
                    timeout=min(0.02, max(0.001, attempt_end - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                pass
        check()
        if process.returncode != 0:
            raise OSError("Provider transport worker failed")
        records = [json.loads(line) for line in output.splitlines()]
        check()
        return records
    finally:
        if process is None:
            _SLOTS.release()
        else:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass  # A concurrent exit or delayed OS kill retains slot.
            # Usually immediate. If the OS cannot reap promptly, retain the
            # slot and bound the detached cleanup by the same four slots.
            try:
                output, _ = process.communicate(timeout=0.1)
            except subprocess.TimeoutExpired:
                threading.Thread(target=_release, args=(process, writer, input_stream), daemon=True).start()
            else:
                if writer is not None:
                    writer.join(timeout=0.02)
                if writer is not None and writer.is_alive():
                    threading.Thread(target=_release, args=(process, writer, input_stream), daemon=True).start()
                else:
                    _release(process, writer, input_stream)
            # Audit only headers already received by this invocation. Never
            # parse/admit a late body or allow a detached reaper to call back.
            # Only safe events/headers are admitted here, including after kill.
            # Late bodies remain isolated, and reapers never call observers.
        if output:
            for line in output.splitlines():
                if not line.startswith((b'{"event":', b'{"http_status":')):
                    continue  # Do not parse late bodies, usage or private output.
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if record.get("attempt_id") == audit["attempt_id"] and "event" in record:
                    if record["event"] == "network_started":
                        audit.update(network_state="started", local_reason="network_api_entered")
                    elif record["event"] == "worker_finished" and record.get("network_state") == "not_started":
                        audit.update(network_state="not_started", local_reason="before_network_failure")
                    observe(record)
                if type(record.get("http_status")) is int:
                    audit.update(network_state="started", local_reason="response_headers")
                    observe({"event": "response_headers", "http_status": record["http_status"]})
                    if on_headers is not None:
                        on_headers(record["http_status"])
        observe({"event": "transport_finished", **audit})
