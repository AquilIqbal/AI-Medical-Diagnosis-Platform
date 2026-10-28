"""Password-hashing utilities for future authentication workflows."""

from werkzeug.security import (
    check_password_hash as werkzeug_check_password_hash,
)
from werkzeug.security import (
    generate_password_hash as werkzeug_generate_password_hash,
)


def generate_password_hash(password: str) -> str:
    """Return Werkzeug's secure hash for a plaintext password."""
    return werkzeug_generate_password_hash(password)


def check_password_hash(password_hash: str, password: str) -> bool:
    """Return whether a plaintext password matches an existing hash."""
    return werkzeug_check_password_hash(password_hash, password)
