"""Focused tests for authenticated account settings."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from backend import create_app
from backend.models.user import User, UserRole
from backend.services.settings_service import SettingsService, SettingsValidationError
from config import Config


class TestConfig(Config):
    """Use controller/service mocks without a database connection."""

    TESTING = True


def _user() -> User:
    user = User(
        id=1,
        full_name="Ada Lovelace",
        username="ada",
        email="ada@example.com",
        role=UserRole.DOCTOR,
        phone_number="555-0101",
        is_active=True,
        last_login=datetime(2026, 9, 1, 12, 30, tzinfo=timezone.utc),
    )
    user.set_password("old-password")
    return user


def _authenticated_client(app):
    client = app.test_client()
    with client.session_transaction() as session:
        session.update(user_id=1, username="ada", role="DOCTOR")
    return client


def test_settings_requires_authentication():
    app = create_app(TestConfig)

    with app.test_client() as client:
        response = client.get("/settings", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login")


def test_settings_route_is_registered_as_auth_settings():
    """The sidebar's ``url_for('auth.settings')`` target remains available."""
    app = create_app(TestConfig)

    settings_rules = [
        rule
        for rule in app.url_map.iter_rules()
        if rule.endpoint == "auth.settings"
    ]

    assert len(settings_rules) == 1
    assert settings_rules[0].rule == "/settings"
    assert "GET" in settings_rules[0].methods


def test_authenticated_user_can_open_settings_and_never_sees_password_hash():
    app = create_app(TestConfig)
    user = _user()

    with patch("backend.controllers.auth_controller.settings_service") as service_mock:
        service_mock.get_user_by_id.return_value = user
        response = _authenticated_client(app).get("/settings")

    assert response.status_code == 200
    assert b"Ada Lovelace" in response.data
    assert b"ada@example.com" in response.data
    assert b"Doctor" in response.data
    assert user.password_hash.encode() not in response.data


def test_profile_fields_are_updated_through_the_settings_endpoint():
    app = create_app(TestConfig)
    user = _user()

    with patch("backend.controllers.auth_controller.settings_service") as service_mock:
        service_mock.update_profile.return_value = user
        response = _authenticated_client(app).post(
            "/settings/profile",
            data={
                "full_name": "Ada Byron",
                "email": "ada.byron@example.com",
                "phone_number": "+1 555 0102",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/settings")
    assert service_mock.update_profile.call_args.args == (
        1,
        {
            "full_name": "Ada Byron",
            "email": "ada.byron@example.com",
            "phone_number": "+1 555 0102",
        },
    )


def test_invalid_profile_input_is_rejected():
    service = SettingsService()
    user = _user()

    with patch.object(service, "get_user_by_id", return_value=user):
        with pytest.raises(SettingsValidationError, match="valid email"):
            service.update_profile(
                1,
                {"full_name": "Ada", "email": "invalid", "phone_number": ""},
            )


def test_profile_service_ignores_username_and_role_changes():
    service = SettingsService()
    user = _user()

    with patch.object(service, "get_user_by_id", return_value=user), patch(
        "backend.services.settings_service.db.session"
    ) as session_mock:
        service.update_profile(
            1,
            {
                "full_name": "Ada Byron",
                "email": "ada.byron@example.com",
                "phone_number": "555-0102",
                "username": "attacker",
                "role": "ADMIN",
            },
        )

    assert user.full_name == "Ada Byron"
    assert user.username == "ada"
    assert user.role is UserRole.DOCTOR
    session_mock.commit.assert_called_once()


def test_password_change_requires_and_verifies_current_password():
    service = SettingsService()
    user = _user()

    with patch.object(service, "get_user_by_id", return_value=user):
        with pytest.raises(SettingsValidationError, match="Current password is required"):
            service.change_password(1, "", "new-password", "new-password")
        with pytest.raises(SettingsValidationError, match="Current password is incorrect"):
            service.change_password(1, "wrong-password", "new-password", "new-password")


def test_password_mismatch_is_rejected():
    service = SettingsService()
    user = _user()

    with patch.object(service, "get_user_by_id", return_value=user):
        with pytest.raises(SettingsValidationError, match="confirmation does not match"):
            service.change_password(1, "old-password", "new-password", "different")


def test_successful_password_change_stores_hash_and_invalidates_old_password():
    service = SettingsService()
    user = _user()
    old_hash = user.password_hash

    with patch.object(service, "get_user_by_id", return_value=user), patch(
        "backend.services.settings_service.db.session"
    ) as session_mock:
        service.change_password(1, "old-password", "new-password", "new-password")

    assert user.password_hash != old_hash
    assert user.password_hash != "new-password"
    assert user.check_password("new-password") is True
    assert user.check_password("old-password") is False
    session_mock.commit.assert_called_once()


def test_password_endpoint_rejects_invalid_current_password():
    app = create_app(TestConfig)
    user = _user()

    with patch("backend.controllers.auth_controller.settings_service") as service_mock:
        service_mock.change_password.side_effect = SettingsValidationError(
            "Current password is incorrect."
        )
        service_mock.get_user_by_id.return_value = user
        response = _authenticated_client(app).post(
            "/settings/password",
            data={
                "current_password": "wrong-password",
                "new_password": "new-password",
                "confirm_password": "new-password",
            },
        )

    assert response.status_code == 400
    assert b"Current password is incorrect." in response.data
