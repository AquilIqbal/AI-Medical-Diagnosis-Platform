"""Reusable SQLAlchemy base model for persisted application entities."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column

from backend.extensions import db


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class BaseModel(db.Model):
    """Abstract base class providing common identity and audit fields.

    Future persisted models can inherit this class to receive a primary key,
    creation timestamp, update timestamp, dictionary serialization, and a
    concise debug representation. As an abstract model, it creates no table.
    """

    __abstract__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize mapped columns into values suitable for API responses."""
        return {
            column.name: self._serialize_value(getattr(self, column.name))
            for column in self.__table__.columns
        }

    def __repr__(self) -> str:
        """Return a concise representation useful during debugging."""
        return f"<{self.__class__.__name__} id={self.id!r}>"

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Convert timestamp values to ISO 8601 strings for serialization."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value
