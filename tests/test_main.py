"""Tests for the public main blueprint."""

from unittest.mock import patch

from backend import create_app
from backend.services.dashboard_service import DashboardStatistics


def test_home_page_renders_platform_name():
    """The root URL returns the initial platform landing page."""
    app = create_app()
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"AI Medical Diagnosis Platform" in response.data


def test_dashboard_displays_available_report_count():
    """The Reports dashboard metric uses the available diagnosis report count."""
    app = create_app()

    with patch("backend.controllers.auth_controller.dashboard_service") as service_mock:
        service_mock.get_statistics.return_value = DashboardStatistics(
            patient_count=2,
            diagnosis_count=5,
            ai_prediction_count=6,
            report_count=5,
        )
        service_mock.get_recent_diagnosis_activity.return_value = []

        with app.test_client() as client:
            with client.session_transaction() as session:
                session.update(user_id=1, username="Test User", role="admin")

            response = client.get("/dashboard")

    assert response.status_code == 200
    assert b"Available diagnosis reports" in response.data
    assert b"Not implemented" not in response.data
    assert b'<strong class="dashboard-stat-card__value">5</strong>' in response.data
    assert b'href="/reports"' in response.data
    assert b">Generate Report</a>" in response.data
