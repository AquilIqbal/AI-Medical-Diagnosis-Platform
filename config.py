"""Environment-aware configuration for the Flask application."""

import os
from urllib.parse import quote_plus

from dotenv import load_dotenv


# Local .env values never override environment values supplied by deployment.
load_dotenv()


def build_database_uri(
    host: str,
    port: int,
    database_name: str,
    username: str,
    password: str,
) -> str:
    """Build a SQLAlchemy MySQL URI with safely encoded credentials."""
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)
    encoded_database_name = quote_plus(database_name)
    return (
        "mysql+pymysql://"
        f"{encoded_username}:{encoded_password}@{host}:{port}/"
        f"{encoded_database_name}?charset=utf8mb4"
    )


class Config:
    """Base configuration shared by all deployment environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "development-only-change-me")
    TEMPLATES_AUTO_RELOAD = True
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # MySQL connection values are sourced exclusively from environment settings.
    DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT = int(os.getenv("DATABASE_PORT", "3306"))
    DATABASE_NAME = os.getenv("DATABASE_NAME", "ai_medical_diagnosis")
    DATABASE_USER = os.getenv("DATABASE_USER", "root")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "")

    SQLALCHEMY_DATABASE_URI = build_database_uri(
        DATABASE_HOST,
        DATABASE_PORT,
        DATABASE_NAME,
        DATABASE_USER,
        DATABASE_PASSWORD,
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024


class DevelopmentConfig(Config):
    """Configuration used during local development."""

    DEBUG = True


class ProductionConfig(Config):
    """Configuration used by production deployments."""

    DEBUG = False
    SECRET_KEY = os.getenv("SECRET_KEY")
    TEMPLATES_AUTO_RELOAD = False
