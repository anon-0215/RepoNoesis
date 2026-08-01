from __future__ import annotations

import uvicorn

from app.config import get_env_value, load_environment


def main() -> None:
    load_environment()
    host = get_env_value("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(get_env_value("BACKEND_PORT", "8000"))
    except ValueError:
        port = 8000
    if port < 1 or port > 65535:
        port = 8000
    uvicorn.run("app.main:app", reload=True, host=host, port=port)


if __name__ == "__main__":
    main()
