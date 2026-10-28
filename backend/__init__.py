"""Flask application package and application factory."""

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from backend.extensions import init_extensions
from backend.models import Diagnosis as _Diagnosis
from backend.models import DiagnosisAIPrediction as _DiagnosisAIPrediction
from backend.models import DiagnosisClinicalAssessment as _DiagnosisClinicalAssessment
from backend.models import Patient as _Patient
from backend.models import User as _User
from backend.routes.auth import auth_bp
from backend.routes.diagnosis import diagnosis_bp
from backend.routes.main import main_bp
from backend.routes.patient import patient_bp
from backend.routes.reports import reports_bp
from backend.utils.logging_config import configure_logging
from config import Config, ProductionConfig


# Importing the model package makes all mapped metadata available to Flask-Migrate.
_ = _Diagnosis, _DiagnosisAIPrediction, _DiagnosisClinicalAssessment, _Patient, _User


def create_app(config_object: type[Config] = Config) -> Flask:
    """Create, configure, and return a Flask application instance."""
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(config_object)

    if config_object is ProductionConfig and not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be configured for production.")

    # Extensions are initialized only after the application's config is loaded.
    init_extensions(app)
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(diagnosis_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(reports_bp)
    configure_logging(app)
    register_error_handlers(app)

    return app


def register_error_handlers(app: Flask) -> None:
    """Register application-wide error pages outside individual blueprints."""

    @app.errorhandler(404)
    def not_found(_error: HTTPException) -> tuple[str, int]:
        """Render a consistent page for unknown URLs."""
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def internal_server_error(error: Exception) -> tuple[str, int]:
        """Render a safe page for unexpected server failures."""
        app.logger.exception("Unhandled server error: %s", error)
        return render_template("errors/500.html"), 500

    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(_error: RequestEntityTooLarge):
        """Return a safe form response when a request exceeds the upload limit."""
        flash("The uploaded request must be 5 MB or smaller.", "error")
        return redirect(request.referrer or url_for("diagnosis.new_diagnosis"))
