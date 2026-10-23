"""Private single-attempt urllib worker. Only stdlib; no app bootstrap or disk IO.

Secrets and source travel through private pipes, never command-line arguments.
The owner kills this process to close sockets during DNS/connect/TLS/header/body
waits. Killing the local process says nothing about remote generation state.
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def main():
    envelope = json.load(sys.stdin)
    def event(name, **values):
        emit({"event": name, "attempt_id": envelope["attempt_id"],
              "monotonic": time.monotonic(), **values})
    def safe_parameters(body):
        if not body or not envelope.get("collect_parameters"):
            return None
        payload = json.loads(body)
        parameters = {key: payload[key] for key in (
            "model", "thinking", "reasoning_effort", "temperature", "max_tokens"
        ) if key in payload}
        parameters["messages_sha256"] = hashlib.sha256(json.dumps(
            payload.get("messages"), ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        return parameters
    parameters = safe_parameters(envelope["body"])
    current_parameters = None
    event("worker_received", parameters=parameters)
    policy = envelope.get("audit_policy")
    started = False
    denied = False

    def audit(name, args):
        nonlocal started, denied, current_parameters
        if name == "urllib.Request":
            # Inspect the body actually handed to urllib.open, after Request
            # construction. A redirected GET has no model/messages body.
            current_parameters = safe_parameters(args[1])
            url = urllib.parse.urlsplit(args[0])
            origin = [url.scheme, url.hostname, url.port or (443 if url.scheme == "https" else 80)]
            allowed = not policy or origin in policy["allowed_origins"]
            if policy and (url.username or url.password or url.scheme not in {"http", "https"}):
                allowed = False
            event("destination", origin=origin, allowed=allowed, parameters=current_parameters)
            if not allowed:
                denied = True
                raise PermissionError("Transport audit destination denied")
        if policy and policy.get("loopback_only"):
            host = None
            if name in {"socket.getaddrinfo", "socket.gethostbyname"}:
                host = args[0]
            elif name == "socket.connect":
                host = args[1][0] if isinstance(args[1], tuple) else "DENIED"
            if host is not None:
                allowed = host in {"127.0.0.1", "::1"}
                event("network_destination", host=host if allowed else "DENIED", allowed=allowed)
                if not allowed:
                    denied = True
                    raise PermissionError("Transport audit network denied")
        # Emitted inside HTTPConnection.connect, before DNS/socket work.
        # This confirms network API entry, never remote receipt or generation.
        if name == "http.client.connect" and not started:
            started = True
            event("network_started", parameters=current_parameters)

    sys.addaudithook(audit)
    request = urllib.request.Request(envelope["url"],
        data=envelope["body"].encode("utf-8"), headers=envelope["headers"], method="POST")
    try:
        with urllib.request.urlopen(request, timeout=envelope["timeout"]) as response:
            emit({"http_status": response.status})
            data = json.loads(response.read().decode("utf-8"))
        emit({"data": data})
    except urllib.error.HTTPError as exc:
        emit({"http_status": exc.code})
        exc.close()
        emit({"error": "http", "status": exc.code})
    except TimeoutError:
        emit({"error": "timeout"})
    except (urllib.error.URLError, OSError):
        emit({"error": "audit" if denied else "network"})
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        emit({"error": "invalid_json"})
    finally:
        event("worker_finished", network_state="started" if started else "not_started")


if __name__ == "__main__":
    main()
