"""Logging setup shared by all application factory instances."""

import logging

from flask import Flask


def configure_logging(app: Flask) -> None:
    """Configure timestamped console logging without duplicate handlers."""
    app.logger.setLevel(app.config["LOG_LEVEL"])

    if app.logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    app.logger.addHandler(handler)
    app.logger.propagate = False
