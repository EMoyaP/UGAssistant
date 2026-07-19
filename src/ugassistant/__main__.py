from __future__ import annotations

import os
import threading

import uvicorn

from ugassistant.api.app import create_app
from ugassistant.config import load_app_settings


def main() -> None:
    settings = load_app_settings()
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            timeout_graceful_shutdown=4,
        )
    )

    def request_shutdown() -> None:
        server.should_exit = True

        # Native camera and audio drivers can occasionally block interpreter
        # shutdown on Windows. The normal Uvicorn lifespan runs first.
        fallback = threading.Timer(8.0, lambda: os._exit(0))
        fallback.daemon = True
        fallback.start()

    app.state.shutdown_callback = request_shutdown
    server.run()


if __name__ == "__main__":
    main()
