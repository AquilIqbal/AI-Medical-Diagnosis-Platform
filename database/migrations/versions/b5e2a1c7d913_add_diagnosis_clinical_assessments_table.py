"""Add diagnosis clinical assessments table.

Revision ID: b5e2a1c7d913
Revises: f4b8c2d9e671
Create Date: 2026-08-28 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "b5e2a1c7d913"
down_revision = "f4b8c2d9e671"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "diagnosis_clinical_assessments",
        sa.Column("diagnosis_id", sa.Integer(), nullable=False),
        sa.Column("smoking", sa.String(length=16), nullable=False),
        sa.Column("bmi", sa.Float(), nullable=False),
        sa.Column("HbA1c_level", sa.Float(), nullable=False),
        sa.Column("glucose", sa.Float(), nullable=False),
        sa.Column("cholesterol", sa.Float(), nullable=False),
        sa.Column("sleep_hours", sa.Float(), nullable=False),
        sa.Column("triglycerides", sa.Float(), nullable=False),
        sa.Column("physical_activity", sa.String(length=16), nullable=False),
        sa.Column("family_history", sa.String(length=8), nullable=False),
        sa.Column("stress_level", sa.String(length=16), nullable=False),
        sa.Column("low_hdl_cholesterol", sa.String(length=8), nullable=False),
        sa.Column("high_ldl_cholesterol", sa.String(length=8), nullable=False),
        sa.Column("blood_pressure", sa.Float(), nullable=False),
        sa.Column("high_blood_pressure", sa.String(length=8), nullable=False),
        sa.Column("sugar_consumption", sa.String(length=16), nullable=False),
        sa.Column("crp_level", sa.Float(), nullable=False),
        sa.Column("homocysteine_level", sa.Float(), nullable=False),
        sa.Column("systolic_bp", sa.Float(), nullable=False),
        sa.Column("diastolic_bp", sa.Float(), nullable=False),
        sa.Column("alcohol_intake", sa.Float(), nullable=False),
        sa.Column("salt_intake", sa.Float(), nullable=False),
        sa.Column("heart_rate", sa.Float(), nullable=False),
        sa.Column("hdl", sa.Float(), nullable=False),
        sa.Column("ldl", sa.Float(), nullable=False),
        sa.Column("education_level", sa.String(length=16), nullable=False),
        sa.Column("employment_status", sa.String(length=16), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["diagnosis_id"], ["diagnoses.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("diagnosis_id"),
    )


def downgrade():
    op.drop_table("diagnosis_clinical_assessments")
