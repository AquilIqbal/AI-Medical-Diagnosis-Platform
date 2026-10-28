"""User persistence model and role definitions."""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Boolean, DateTime, Enum as SqlEnum, String
from sqlalchemy.orm import Mapped, mapped_column
from werkzeug.security import check_password_hash, generate_password_hash

from backend.models.base import BaseModel


class UserRole(str, Enum):
    """Roles available to future platform users."""

    ADMIN = "ADMIN"
    DOCTOR = "DOCTOR"
    RECEPTIONIST = "RECEPTIONIST"


class User(BaseModel):
    """Represents a platform user with role and profile information.

    This model defines persistence and password-hash utilities only. It does
    not provide authentication, session, or login behavior.
    """

    __tablename__ = "users"

    full_name: Mapped[str] = mapped_column(String(150), nullable=False)
    username: Mapped[str] = mapped_column(
        String(80),
        unique=True,
        index=True,
        nullable=False,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        SqlEnum(UserRole, name="user_role"),
        default=UserRole.RECEPTIONIST,
        index=True,
        nullable=False,
    )
    phone_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    profile_image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        index=True,
        nullable=False,
    )
    last_login: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def set_password(self, password: str) -> None:
        """Hash ``password`` and store only its secure derived value."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Return whether ``password`` matches the stored password hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self) -> dict[str, Any]:
        """Serialize public user data without exposing the password hash."""
        user_data = super().to_dict()
        user_data.pop("password_hash", None)
        user_data["role"] = self.role.value
        return user_data
