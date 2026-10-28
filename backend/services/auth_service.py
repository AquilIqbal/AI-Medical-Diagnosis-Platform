"""Authentication business logic for credential-based access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from flask import current_app
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError

from backend.extensions import db
from backend.models.user import User
from backend.utils.password import check_password_hash


class AuthenticationStatus(str, Enum):
    """Outcomes returned by credential authentication attempts."""

    SUCCESS = "success"
    INVALID_CREDENTIALS = "invalid_credentials"
    INACTIVE_ACCOUNT = "inactive_account"
    ERROR = "error"


@dataclass(frozen=True)
class AuthenticationResult:
    """Result of an authentication attempt without HTTP concerns."""

    status: AuthenticationStatus
    user: User | None = None


class AuthService:
    """Coordinates credential validation and account state updates."""

    def login(
        self,
        username_or_email: str,
        password: str,
    ) -> AuthenticationResult:
        """Authenticate credentials for a future or current login request."""
        return self.authenticate_user(username_or_email, password)

    def logout(self) -> None:
        """Provide a service-layer extension point for future logout auditing."""

    def authenticate_user(
        self,
        username_or_email: str,
        password: str,
    ) -> AuthenticationResult:
        """Verify credentials, reject inactive users, and record last login."""
        identifier = username_or_email.strip()
        if not identifier or not password:
            return AuthenticationResult(AuthenticationStatus.INVALID_CREDENTIALS)

        try:
            user = db.session.scalar(
                select(User).where(
                    or_(
                        User.username == identifier,
                        User.email == identifier.lower(),
                    )
                )
            )
        except SQLAlchemyError:
            db.session.rollback()
            current_app.logger.exception("Unable to retrieve user credentials.")
            return AuthenticationResult(AuthenticationStatus.ERROR)

        if user is None:
            return AuthenticationResult(AuthenticationStatus.INVALID_CREDENTIALS)

        try:
            password_matches = check_password_hash(user.password_hash, password)
        except ValueError:
            current_app.logger.warning(
                "Password hash validation failed for user id %s.",
                user.id,
            )
            return AuthenticationResult(AuthenticationStatus.INVALID_CREDENTIALS)

        if not password_matches:
            return AuthenticationResult(AuthenticationStatus.INVALID_CREDENTIALS)

        if not user.is_active:
            return AuthenticationResult(AuthenticationStatus.INACTIVE_ACCOUNT)

        try:
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            current_app.logger.exception("Unable to update the user's last login.")
            return AuthenticationResult(AuthenticationStatus.ERROR)

        return AuthenticationResult(AuthenticationStatus.SUCCESS, user)
