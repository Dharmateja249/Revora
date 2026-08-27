"""
Revora Historical Dataset Ingestion Module.

Loads verified historical recovery records from CSV into Revora's relational
database models (Customer, Payment, RecoveryOpportunity, RecoveryAttempt, AuditEvent)
with strict schema validation, deduplication, and idempotent execution.
"""

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set
import uuid

# Ensure backend root is in sys.path when executed directly as a script
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import (
    Customer,
    Payment,
    RecoveryOpportunity,
    RecoveryAttempt,
    AuditEvent,
    utc_now,
)

logger = logging.getLogger("revora.historical_data")

# Complete list of mandatory columns expected in the historical CSV dataset
EXPECTED_CSV_COLUMNS: List[str] = [
    "record_id",
    "customer_id",
    "payment_id",
    "customer_payment_count",
    "customer_success_rate",
    "customer_previous_failures",
    "payment_amount",
    "currency",
    "payment_method",
    "failure_reason",
    "attempt_number",
    "hours_since_failure",
    "action_taken",
    "previous_action",
    "previous_attempt_count",
    "recovered",
    "amount_recovered",
    "recovery_time_hours",
]


def validate_dataset_schema(filepath: str | Path) -> None:
    """
    Validate that the CSV file exists and contains all required columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV header is missing required columns.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Historical dataset not found at path: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Historical dataset file is empty: {path}")

    header_clean = [col.strip() for col in header]
    missing_columns = [col for col in EXPECTED_CSV_COLUMNS if col not in header_clean]

    if missing_columns:
        raise ValueError(
            f"Historical dataset schema validation failed. Missing required columns: {missing_columns}"
        )


def _parse_bool(val: Any) -> bool:
    """Parse boolean value from various string/bool representations."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "t", "yes", "y")


def load_historical_data(
    db_session: Session,
    csv_path: str | Path = "data/historical_recovery_data.csv",
    batch_commit_size: int = 500,
) -> Dict[str, int]:
    """
    Ingest verified historical recovery data from a CSV file into the database.

    Features:
    - Schema validation prior to ingestion
    - Entity deduplication (Customer, Payment, RecoveryOpportunity)
    - Attempt deduplication via external_reference (record_id) for safe re-runs
    - Deterministic mapping preserving relational integrity
    - Domain rule enforcement: only recovered=True creates succeeded recovery outcomes

    Args:
        db_session: Active SQLAlchemy database session.
        csv_path: Path to historical CSV dataset.
        batch_commit_size: Periodic commit batch size.

    Returns:
        Dict with summary counts of created and processed entities.
    """
    path = Path(csv_path)
    validate_dataset_schema(path)

    # In-memory tracking caches to avoid repetitive database lookups and prevent duplicates
    # Pre-populate with existing records from the database for full idempotency across runs
    existing_customers: Dict[str, Customer] = {
        c.external_customer_id: c
        for c in db_session.query(Customer).filter(Customer.external_customer_id.isnot(None)).all()
    }

    existing_payments: Dict[str, Payment] = {
        p.external_payment_id: p
        for p in db_session.query(Payment).filter(Payment.external_payment_id.isnot(None)).all()
    }

    existing_attempts_by_ref: Set[str] = {
        a.external_reference
        for a in db_session.query(RecoveryAttempt.external_reference)
        .filter(RecoveryAttempt.external_reference.isnot(None))
        .all()
    }

    counts = {
        "total_records_processed": 0,
        "customers_created": 0,
        "payments_created": 0,
        "opportunities_created": 0,
        "attempts_created": 0,
        "audit_events_created": 0,
        "skipped_duplicate_attempts": 0,
    }

    # Read and parse all CSV rows
    records: List[Dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    # Sort deterministically by payment_id and attempt_number
    def sort_key(r: Dict[str, Any]):
        return (r["payment_id"], int(r.get("attempt_number", 1)))

    records.sort(key=sort_key)

    for row in records:
        counts["total_records_processed"] += 1

        rec_id = row["record_id"].strip()
        cust_id = row["customer_id"].strip()
        pay_id = row["payment_id"].strip()

        # Parse numeric and boolean fields
        payment_amount = float(row["payment_amount"])
        currency = row.get("currency", "INR").strip()
        payment_method = row["payment_method"].strip()
        failure_reason = row["failure_reason"].strip()
        attempt_number = int(row["attempt_number"])
        hours_since_failure = float(row["hours_since_failure"])
        action_taken = row["action_taken"].strip()
        is_recovered = _parse_bool(row["recovered"])
        amount_recovered = float(row["amount_recovered"]) if is_recovered else 0.0
        recovery_time_hours = float(row.get("recovery_time_hours", 0.0))

        cust_payment_count = int(row["customer_payment_count"])
        cust_success_rate = float(row["customer_success_rate"])
        cust_prev_failures = int(row["customer_previous_failures"])
        cust_successful_payments = max(0, cust_payment_count - cust_prev_failures)

        # 1. Customer Resolution / Creation
        if cust_id in existing_customers:
            customer = existing_customers[cust_id]
        else:
            customer = Customer(
                external_customer_id=cust_id,
                name=f"Customer {cust_id}",
                email=f"{cust_id}@revora-demo.internal",
                total_payments=cust_payment_count,
                successful_payments=cust_successful_payments,
                failed_payments=cust_prev_failures,
            )
            db_session.add(customer)
            existing_customers[cust_id] = customer
            counts["customers_created"] += 1

        # 2. Payment Resolution / Creation
        if pay_id in existing_payments:
            payment = existing_payments[pay_id]
            # If this subsequent attempt succeeded, update payment status to succeeded
            if is_recovered and payment.status != "succeeded":
                payment.status = "succeeded"
        else:
            # Payment status: 'succeeded' if this attempt recovered, else 'failed'
            initial_status = "succeeded" if is_recovered else "failed"
            payment = Payment(
                external_payment_id=pay_id,
                customer=customer,
                amount=payment_amount,
                currency=currency,
                payment_method=payment_method,
                status=initial_status,
                failure_reason=failure_reason,
            )
            db_session.add(payment)
            existing_payments[pay_id] = payment
            counts["payments_created"] += 1

        # 3. RecoveryOpportunity Resolution / Creation (1:1 with Payment)
        if payment.recovery_opportunity is not None:
            opportunity = payment.recovery_opportunity
            # Update opportunity outcome if recovered
            if is_recovered:
                opportunity.status = "recovered"
                opportunity.expected_recovery = amount_recovered
        else:
            opp_status = "recovered" if is_recovered else "failed"
            opportunity = RecoveryOpportunity(
                payment=payment,
                status=opp_status,
                revenue_at_risk=payment_amount,
                expected_recovery=amount_recovered if is_recovered else 0.0,
                recommended_action=action_taken,
                confidence=cust_success_rate,
            )
            db_session.add(opportunity)
            counts["opportunities_created"] += 1

        # 4. RecoveryAttempt Creation (Idempotent check)
        if rec_id in existing_attempts_by_ref:
            counts["skipped_duplicate_attempts"] += 1
            continue

        attempt_status = "succeeded" if is_recovered else "failed"
        attempt = RecoveryAttempt(
            opportunity=opportunity,
            action=action_taken,
            status=attempt_status,
            amount_recovered=amount_recovered,
            external_reference=rec_id,
            error_code=failure_reason if not is_recovered else None,
        )
        db_session.add(attempt)
        existing_attempts_by_ref.add(rec_id)
        counts["attempts_created"] += 1

        # 5. AuditEvent Creation for Historical Verification Tracking
        audit_event = AuditEvent(
            opportunity=opportunity,
            event_type="historical_outcome_ingested",
            description=(
                f"Ingested verified historical attempt #{attempt_number} ({action_taken}) "
                f"with outcome: {attempt_status} (recovered={is_recovered})."
            ),
            metadata_payload={
                "record_id": rec_id,
                "attempt_number": attempt_number,
                "action_taken": action_taken,
                "recovered": is_recovered,
                "amount_recovered": amount_recovered,
                "recovery_time_hours": recovery_time_hours,
                "hours_since_failure": hours_since_failure,
                "customer_success_rate": cust_success_rate,
                "failure_reason": failure_reason,
            },
        )
        db_session.add(audit_event)
        counts["audit_events_created"] += 1

        # Periodic commit for batch efficiency
        if counts["attempts_created"] % batch_commit_size == 0:
            db_session.flush()

    db_session.commit()
    logger.info("Historical data ingestion completed successfully: %s", counts)
    return counts


def main():
    """CLI entrypoint for running historical data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest Revora synthetic historical recovery dataset into the database."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/historical_recovery_data.csv",
        help="Path to the historical recovery dataset CSV (default: data/historical_recovery_data.csv)",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database tables before ingestion.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.init_db:
        logger.info("Initializing database schema...")
        init_db()

    csv_file = Path(args.csv_path)
    if not csv_file.is_absolute():
        # Resolve relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        csv_file = repo_root / args.csv_path

    logger.info(f"Loading historical data from: {csv_file}")
    with SessionLocal() as db:
        summary = load_historical_data(db, csv_path=csv_file)
        print("\n" + "=" * 60)
        print("REVORA HISTORICAL DATA INGESTION SUMMARY")
        print("=" * 60)
        for key, val in summary.items():
            print(f"  {key:<30}: {val}")
        print("=" * 60)


if __name__ == "__main__":
    main()
