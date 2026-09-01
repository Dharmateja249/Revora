"""Initial schema creation for Revora domain entities.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-27 22:20:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. customers table
    op.create_table(
        "customers",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("external_customer_id", sa.String(255), nullable=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("total_payments", sa.Integer(), default=0, nullable=False),
        sa.Column("successful_payments", sa.Integer(), default=0, nullable=False),
        sa.Column("failed_payments", sa.Integer(), default=0, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_customers_email", "customers", ["email"])
    op.create_index(
        "ix_customers_external_customer_id",
        "customers",
        ["external_customer_id"],
        unique=True,
    )

    # 2. payments table
    op.create_table(
        "payments",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("external_payment_id", sa.String(255), nullable=True),
        sa.Column(
            "customer_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("customers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(10), default="INR", nullable=False),
        sa.Column("payment_method", sa.String(50), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("failure_reason", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_payments_customer_id", "payments", ["customer_id"])
    op.create_index(
        "ix_payments_external_payment_id",
        "payments",
        ["external_payment_id"],
        unique=True,
    )
    op.create_index("ix_payments_status", "payments", ["status"])

    # 3. recovery_opportunities table
    op.create_table(
        "recovery_opportunities",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "payment_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("payments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("revenue_at_risk", sa.Float(), nullable=False),
        sa.Column("expected_recovery", sa.Float(), nullable=False),
        sa.Column("recommended_action", sa.String(100), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_recovery_opportunities_payment_id",
        "recovery_opportunities",
        ["payment_id"],
        unique=True,
    )
    op.create_index(
        "ix_recovery_opportunities_status", "recovery_opportunities", ["status"]
    )

    # 4. recovery_attempts table
    op.create_table(
        "recovery_attempts",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "opportunity_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("recovery_opportunities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("status", sa.String(50), nullable=False),
        sa.Column("amount_recovered", sa.Float(), default=0.0, nullable=False),
        sa.Column("external_reference", sa.String(255), nullable=True),
        sa.Column("error_code", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_recovery_attempts_opportunity_id", "recovery_attempts", ["opportunity_id"]
    )
    op.create_index("ix_recovery_attempts_status", "recovery_attempts", ["status"])

    # 5. audit_events table
    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "opportunity_id",
            sa.Uuid(as_uuid=True),
            sa.ForeignKey("recovery_opportunities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index(
        "ix_audit_events_opportunity_id", "audit_events", ["opportunity_id"]
    )


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("recovery_attempts")
    op.drop_table("recovery_opportunities")
    op.drop_table("payments")
    op.drop_table("customers")
