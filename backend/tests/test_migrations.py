import pytest
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError, DBAPIError
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command

from app.database import Base
from app.models import (
    Customer,
    Payment,
    RecoveryOpportunity,
    RecoveryAttempt,
    AuditEvent,
)
from app.historical_data import load_historical_data


@pytest.fixture
def alembic_config_factory(tmp_path):
    """Factory fixture returning Alembic Config and db_url for a new temporary database."""
    counter = 0

    def _create_config():
        nonlocal counter
        counter += 1
        db_file = tmp_path / f"test_migration_{counter}.db"
        db_url = f"sqlite:///{db_file}"

        alembic_ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        alembic_cfg.set_main_option(
            "script_location", str(Path(__file__).resolve().parent.parent / "alembic")
        )
        return alembic_cfg, db_url

    return _create_config


def test_fresh_database_upgrade_and_downgrade(alembic_config_factory):
    """
    Verify that on a fresh empty database, `alembic upgrade head` creates all
    tables and indexes, and `downgrade base` cleanly removes them.
    """
    cfg, db_url = alembic_config_factory()

    # Upgrade to head
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    expected_tables = {
        "customers",
        "payments",
        "recovery_opportunities",
        "recovery_attempts",
        "audit_events",
    }
    assert expected_tables.issubset(tables)

    # Verify unique index exists on recovery_attempts.external_reference
    indexes = inspector.get_indexes("recovery_attempts")
    index_names = {idx["name"] for idx in indexes}
    assert "ix_recovery_attempts_external_reference" in index_names

    # Downgrade to base
    command.downgrade(cfg, "base")

    inspector = inspect(engine)
    tables_after = set(inspector.get_table_names())
    assert "recovery_attempts" not in tables_after
    assert "customers" not in tables_after


def test_preexisting_unmanaged_table_fails_initial_migration(alembic_config_factory):
    """
    Verify that if a database has pre-existing tables but no Alembic revision history,
    `alembic upgrade head` fails instead of silently adopting the pre-existing table,
    and the pre-existing unmanaged data is not deleted.
    """
    cfg, db_url = alembic_config_factory()
    engine = create_engine(db_url)

    # Simulate an unmanaged database with pre-existing 'customers' table and data
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE customers ("
                "id VARCHAR(36) PRIMARY KEY, "
                "name VARCHAR(255) NOT NULL, "
                "email VARCHAR(255) NOT NULL"
                ")"
            )
        )
        conn.execute(
            text(
                "INSERT INTO customers (id, name, email) "
                "VALUES ('cust_unmanaged_1', 'Legacy User', 'legacy@example.com')"
            )
        )

    # Attempting to run initial migration MUST fail because 'customers' already exists
    with pytest.raises((OperationalError, DBAPIError, Exception)):
        command.upgrade(cfg, "head")

    # Verify pre-existing unmanaged data was NOT dropped or deleted
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT name, email FROM customers WHERE id = 'cust_unmanaged_1'")
        ).fetchone()
        assert row is not None
        assert row[0] == "Legacy User"
        assert row[1] == "legacy@example.com"


def test_migration_0002_cleans_legacy_duplicates(alembic_config_factory):
    """
    Verify that migration 0002 safely resolves duplicate non-null external_reference
    records and applies the unique constraint.
    """
    cfg, db_url = alembic_config_factory()

    # 1. Upgrade to initial schema (0001)
    command.upgrade(cfg, "0001_initial_schema")

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO customers (id, name, email, total_payments, successful_payments, failed_payments, created_at) "
                "VALUES ('00000000-0000-0000-0000-000000000001', 'Test Customer', 'test@example.com', 1, 1, 0, '2026-08-27 10:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO payments (id, customer_id, amount, currency, payment_method, status, created_at, updated_at) "
                "VALUES ('00000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000001', 100.0, 'INR', 'card', 'failed', '2026-08-27 10:00:00', '2026-08-27 10:00:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO recovery_opportunities (id, payment_id, status, revenue_at_risk, expected_recovery, created_at, updated_at) "
                "VALUES ('00000000-0000-0000-0000-000000000003', '00000000-0000-0000-0000-000000000002', 'open', 100.0, 0.0, '2026-08-27 10:00:00', '2026-08-27 10:00:00')"
            )
        )

        # Insert 3 duplicate recovery attempts with identical external_reference
        conn.execute(
            text(
                "INSERT INTO recovery_attempts (id, opportunity_id, action, status, amount_recovered, external_reference, created_at) "
                "VALUES ('00000000-0000-0000-0000-000000000011', '00000000-0000-0000-0000-000000000003', 'RETRY', 'failed', 0.0, 'rec_legacy_dup', '2026-08-27 10:01:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO recovery_attempts (id, opportunity_id, action, status, amount_recovered, external_reference, created_at) "
                "VALUES ('00000000-0000-0000-0000-000000000012', '00000000-0000-0000-0000-000000000003', 'PAYMENT_LINK', 'failed', 0.0, 'rec_legacy_dup', '2026-08-27 10:02:00')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO recovery_attempts (id, opportunity_id, action, status, amount_recovered, external_reference, created_at) "
                "VALUES ('00000000-0000-0000-0000-000000000013', '00000000-0000-0000-0000-000000000003', 'REMINDER', 'failed', 0.0, 'rec_legacy_dup', '2026-08-27 10:03:00')"
            )
        )

    # 2. Run migration 0002 to head
    command.upgrade(cfg, "head")

    # 3. Verify exactly 1 attempt remains (earliest)
    with engine.connect() as conn:
        remaining = conn.execute(
            text("SELECT id, action FROM recovery_attempts WHERE external_reference = 'rec_legacy_dup'")
        ).fetchall()
        assert len(remaining) == 1
        assert remaining[0][0] in ("00000000-0000-0000-0000-000000000011", "00000000000000000000000000000011")
        assert remaining[0][1] == "RETRY"

    # 4. Verify duplicate insert fails
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO recovery_attempts (id, opportunity_id, action, status, amount_recovered, external_reference, created_at) "
                    "VALUES ('00000000-0000-0000-0000-000000000099', '00000000-0000-0000-0000-000000000003', 'STOP', 'failed', 0.0, 'rec_legacy_dup', '2026-08-27 10:04:00')"
                )
            )


def test_multiple_null_external_reference_allowed(alembic_config_factory):
    """
    Verify that multiple NULL external_reference values can coexist without constraint violation.
    """
    cfg, db_url = alembic_config_factory()
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    customer = Customer(name="Null Ref Customer", email="nullref@example.com")
    session.add(customer)
    session.commit()

    payment = Payment(
        customer_id=customer.id,
        amount=500.0,
        currency="INR",
        payment_method="upi",
        status="failed",
    )
    session.add(payment)
    session.commit()

    opp = RecoveryOpportunity(
        payment_id=payment.id,
        status="open",
        revenue_at_risk=500.0,
        expected_recovery=0.0,
    )
    session.add(opp)
    session.commit()

    att1 = RecoveryAttempt(
        opportunity_id=opp.id,
        action="RETRY",
        status="failed",
        external_reference=None,
    )
    att2 = RecoveryAttempt(
        opportunity_id=opp.id,
        action="PAYMENT_LINK",
        status="failed",
        external_reference=None,
    )
    session.add_all([att1, att2])
    session.commit()

    null_count = (
        session.query(RecoveryAttempt)
        .filter(RecoveryAttempt.opportunity_id == opp.id, RecoveryAttempt.external_reference.is_(None))
        .count()
    )
    assert null_count == 2
    session.close()


def test_migrated_database_historical_ingestion(alembic_config_factory):
    """
    Verify that a database created and migrated via Alembic ingests the full
    historical dataset successfully and remains idempotent.
    """
    cfg, db_url = alembic_config_factory()
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "historical_recovery_data.csv"
    if not csv_path.exists():
        pytest.skip("Historical dataset CSV not found.")

    summary_1 = load_historical_data(session, csv_path=csv_path)
    assert summary_1["total_records_processed"] == 5000
    assert summary_1["customers_created"] == 1478
    assert summary_1["payments_created"] == 3656
    assert summary_1["opportunities_created"] == 3656
    assert summary_1["attempts_created"] == 5000
    assert summary_1["audit_events_created"] == 5000

    # Second run produces zero duplicates
    summary_2 = load_historical_data(session, csv_path=csv_path)
    assert summary_2["customers_created"] == 0
    assert summary_2["payments_created"] == 0
    assert summary_2["opportunities_created"] == 0
    assert summary_2["attempts_created"] == 0
    assert summary_2["audit_events_created"] == 0
    assert summary_2["skipped_duplicate_attempts"] == 5000

    session.close()
