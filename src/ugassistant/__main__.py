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
    mobile_server: uvicorn.Server | None = None
    mobile_store = app.state.mobile_access_store
    if (
        mobile_store.has_active_access()
        and settings.mobile_tls_certificate_path.is_file()
        and settings.mobile_tls_key_path.is_file()
    ):
        mobile_server = uvicorn.Server(
            uvicorn.Config(
                app.state.mobile_app,
                host="0.0.0.0",
                port=settings.mobile_port,
                ssl_certfile=str(settings.mobile_tls_certificate_path),
                ssl_keyfile=str(settings.mobile_tls_key_path),
                timeout_graceful_shutdown=4,
            )
        )
        threading.Thread(target=mobile_server.run, name="ugassistant-mobile", daemon=True).start()

    def request_shutdown() -> None:
        server.should_exit = True
        if mobile_server is not None:
            mobile_server.should_exit = True

        # Native camera and audio drivers can occasionally block interpreter
        # shutdown on Windows. The normal Uvicorn lifespan runs first.
        fallback = threading.Timer(8.0, lambda: os._exit(0))
        fallback.daemon = True
        fallback.start()

    app.state.shutdown_callback = request_shutdown
    server.run()


if __name__ == "__main__":
    main()
