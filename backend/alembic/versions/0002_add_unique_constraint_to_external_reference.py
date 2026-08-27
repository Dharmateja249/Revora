"""Add unique constraint and index for RecoveryAttempt.external_reference.

Revision ID: 0002_add_unique_external_reference
Revises: 0001_initial_schema
Create Date: 2026-08-27 22:25:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "0002_add_unique_external_reference"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "recovery_attempts" not in inspector.get_table_names():
        return

    # 1. Safely resolve any existing duplicate non-null external_reference records before applying constraint
    # Group by external_reference where non-null having count > 1
    duplicates_query = sa.text(
        "SELECT external_reference FROM recovery_attempts "
        "WHERE external_reference IS NOT NULL "
        "GROUP BY external_reference "
        "HAVING COUNT(*) > 1"
    )
    duplicates = bind.execute(duplicates_query).fetchall()

    for dup_row in duplicates:
        ext_ref = dup_row[0]
        # Query all records for this external_reference ordered by created_at, id
        records_query = sa.text(
            "SELECT id FROM recovery_attempts "
            "WHERE external_reference = :ext_ref "
            "ORDER BY created_at ASC, id ASC"
        )
        records = bind.execute(records_query, {"ext_ref": ext_ref}).fetchall()

        if len(records) > 1:
            # Preserve the first valid record, delete subsequent duplicates
            redundant_ids = [r[0] for r in records[1:]]
            for redundant_id in redundant_ids:
                bind.execute(
                    sa.text("DELETE FROM recovery_attempts WHERE id = :dup_id"),
                    {"dup_id": redundant_id},
                )

    # 2. Re-inspect indexes and add the UNIQUE index/constraint if not already present
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("recovery_attempts")}
    if "ix_recovery_attempts_external_reference" not in existing_indexes:
        with op.batch_alter_table("recovery_attempts") as batch_op:
            batch_op.create_index(
                "ix_recovery_attempts_external_reference",
                ["external_reference"],
                unique=True,
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if "recovery_attempts" in inspector.get_table_names():
        existing_indexes = {idx["name"] for idx in inspector.get_indexes("recovery_attempts")}
        if "ix_recovery_attempts_external_reference" in existing_indexes:
            with op.batch_alter_table("recovery_attempts") as batch_op:
                batch_op.drop_index("ix_recovery_attempts_external_reference")
