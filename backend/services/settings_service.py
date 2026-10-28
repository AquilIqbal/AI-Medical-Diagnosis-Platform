"""Current-user account settings operations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.extensions import db
from backend.models.user import User


class SettingsValidationError(ValueError):
    """Raised when submitted account settings are invalid."""


class SettingsService:
    """Safely manages the authenticated user's editable account settings."""

    _PROFILE_FIELDS = frozenset({"full_name", "email", "phone_number"})
    _EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    _PHONE_PATTERN = re.compile(r"^[0-9+().\-\s]{7,32}$")

    def get_user_by_id(self, user_id: int) -> User | None:
        """Return the authenticated user, if the account still exists."""
        return db.session.get(User, user_id)

    def update_profile(self, user_id: int, form_data: Mapping[str, Any]) -> User | None:
        """Validate and persist only the permitted profile fields."""
        user = self.get_user_by_id(user_id)
        if user is None:
            return None

        profile_data = self._clean_profile_data(form_data)
        for field, value in profile_data.items():
            setattr(user, field, value)

        try:
            db.session.commit()
        except IntegrityError as error:
            db.session.rollback()
            raise SettingsValidationError(
                "That email address is already in use."
            ) from error
        except SQLAlchemyError as error:
            db.session.rollback()
            raise SettingsValidationError(
                "Unable to update account information."
            ) from error

        return user

    def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str,
        confirm_password: str,
    ) -> User | None:
        """Verify the current password before storing a newly hashed password."""
        user = self.get_user_by_id(user_id)
        if user is None:
            return None

        if not current_password:
            raise SettingsValidationError("Current password is required.")
        if not user.check_password(current_password):
            raise SettingsValidationError("Current password is incorrect.")
        if not new_password:
            raise SettingsValidationError("New password is required.")
        if new_password != confirm_password:
            raise SettingsValidationError("New password confirmation does not match.")

        user.set_password(new_password)
        try:
            db.session.commit()
        except SQLAlchemyError as error:
            db.session.rollback()
            raise SettingsValidationError("Unable to change password.") from error

        return user

    def _clean_profile_data(self, form_data: Mapping[str, Any]) -> dict[str, str | None]:
        """Normalize and validate fields that users may change themselves."""
        profile_data = {
            field: self._normalize_value(form_data.get(field))
            for field in self._PROFILE_FIELDS
        }

        full_name = profile_data["full_name"]
        if not full_name:
            raise SettingsValidationError("Full name is required.")
        if len(full_name) > 150:
            raise SettingsValidationError("Full name must be 150 characters or fewer.")

        email = profile_data["email"]
        if not email or not self._EMAIL_PATTERN.fullmatch(email):
            raise SettingsValidationError("Enter a valid email address.")
        if len(email) > 255:
            raise SettingsValidationError("Email must be 255 characters or fewer.")
        profile_data["email"] = email.lower()

        phone_number = profile_data["phone_number"]
        if phone_number and not self._PHONE_PATTERN.fullmatch(phone_number):
            raise SettingsValidationError("Enter a valid phone number.")

        return profile_data

    @staticmethod
    def _normalize_value(value: Any) -> str | None:
        """Trim a submitted value and represent blanks as ``None``."""
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None
