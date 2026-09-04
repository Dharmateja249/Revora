"""Add idempotency_key column and unique index to RecoveryAttempt.

Revision ID: 0003_add_idempotency_key
Revises: 0002_add_unique_external_reference
Create Date: 2026-09-03 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "0003_add_idempotency_key"
down_revision: str | None = "0002_add_unique_external_reference"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "recovery_attempts" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("recovery_attempts")}
    if "idempotency_key" not in columns:
        with op.batch_alter_table("recovery_attempts") as batch_op:
            batch_op.add_column(
                sa.Column("idempotency_key", sa.String(length=255), nullable=True)
            )

    existing_indexes = {
        idx["name"] for idx in inspector.get_indexes("recovery_attempts")
    }
    if "ix_recovery_attempts_idempotency_key" not in existing_indexes:
        with op.batch_alter_table("recovery_attempts") as batch_op:
            batch_op.create_index(
                "ix_recovery_attempts_idempotency_key",
                ["idempotency_key"],
                unique=True,
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "recovery_attempts" in inspector.get_table_names():
        existing_indexes = {
            idx["name"] for idx in inspector.get_indexes("recovery_attempts")
        }
        if "ix_recovery_attempts_idempotency_key" in existing_indexes:
            with op.batch_alter_table("recovery_attempts") as batch_op:
                batch_op.drop_index("ix_recovery_attempts_idempotency_key")

        columns = {col["name"] for col in inspector.get_columns("recovery_attempts")}
        if "idempotency_key" in columns:
            with op.batch_alter_table("recovery_attempts") as batch_op:
                batch_op.drop_column("idempotency_key")
