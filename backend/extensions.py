"""Shared Flask extension instances and their application initialization."""

from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy


# Extensions remain unbound here so the application factory can initialize
# independent app instances for development, tests, and production.
db = SQLAlchemy()
migrate = Migrate()


def init_extensions(app: Flask) -> None:
    """Bind configured SQLAlchemy and Flask-Migrate extensions to ``app``."""
    db.init_app(app)
    migrate.init_app(app, db, compare_type=True)
