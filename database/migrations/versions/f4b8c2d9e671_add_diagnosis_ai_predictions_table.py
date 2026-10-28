"""Add diagnosis AI predictions table.

Revision ID: f4b8c2d9e671
Revises: adccdda8fbce
Create Date: 2026-08-25 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f4b8c2d9e671"
down_revision = "adccdda8fbce"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "diagnosis_ai_predictions",
        sa.Column("diagnosis_id", sa.Integer(), nullable=False),
        sa.Column("prediction_type", sa.String(length=32), nullable=False),
        sa.Column("predicted_label", sa.String(length=255), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("visualization_path", sa.String(length=255), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["diagnosis_id"],
            ["diagnoses.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "diagnosis_id",
            "prediction_type",
            name="uq_diagnosis_ai_predictions_diagnosis_prediction_type",
        ),
    )
    with op.batch_alter_table("diagnosis_ai_predictions", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_diagnosis_ai_predictions_prediction_type"),
            ["prediction_type"],
            unique=False,
        )


def downgrade():
    with op.batch_alter_table("diagnosis_ai_predictions", schema=None) as batch_op:
        batch_op.drop_index(
            batch_op.f("ix_diagnosis_ai_predictions_prediction_type")
        )

    op.drop_table("diagnosis_ai_predictions")
