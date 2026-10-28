"""Schema tests for diagnosis AI prediction persistence."""

from sqlalchemy import inspect

from backend.models.diagnosis import Diagnosis
from backend.models.diagnosis_ai_prediction import DiagnosisAIPrediction


def test_diagnosis_ai_prediction_schema_and_relationship() -> None:
    """The prediction table supports one result per model modality."""
    table = DiagnosisAIPrediction.__table__

    assert set(table.columns.keys()) == {
        "id",
        "diagnosis_id",
        "prediction_type",
        "predicted_label",
        "confidence_score",
        "model_name",
        "visualization_path",
        "created_at",
        "updated_at",
    }
    assert table.c.visualization_path.nullable is True
    assert table.c.predicted_label.nullable is False
    assert table.c.confidence_score.nullable is False

    foreign_key = next(iter(table.foreign_keys))
    assert foreign_key.target_fullname == "diagnoses.id"
    assert foreign_key.ondelete == "CASCADE"
    assert any(
        constraint.name
        == "uq_diagnosis_ai_predictions_diagnosis_prediction_type"
        for constraint in table.constraints
    )

    relationship = inspect(Diagnosis).relationships["ai_predictions"]
    assert relationship.uselist is True
    assert relationship.back_populates == "diagnosis"
    assert relationship.cascade.delete_orphan
    assert relationship.passive_deletes is True
