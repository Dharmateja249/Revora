"""
Revora Synthetic Recovery-History Dataset Generator.

Generates a reproducible, probabilistic historical dataset representing failed
payment recovery attempts and their outcomes. This serves as historical recovery
memory for Revora's intelligence layer.
"""

import csv
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Controlled domain sets
FAILURE_REASONS = [
    "bank_timeout",
    "insufficient_funds",
    "authentication_failed",
    "network_error",
    "technical_error",
    "payment_method_issue",
    "unknown",
]

PAYMENT_METHODS = [
    "card",
    "upi",
    "netbanking",
    "wallet",
]

RECOVERY_ACTIONS = [
    "RETRY",
    "PAYMENT_LINK",
    "REMINDER",
    "ESCALATE",
    "STOP",
]

# Baseline recovery probability by (failure_reason, action_taken)
# Reflects realistic payment domain behaviors:
# - bank_timeout / network_error: RETRY is highly effective.
# - insufficient_funds: PAYMENT_LINK / REMINDER outperform immediate RETRY.
# - authentication_failed / payment_method_issue: Customer intervention (PAYMENT_LINK) outperforms RETRY.
# - STOP: Always 0.0.
BASE_RECOVERY_PROBABILITIES: dict[str, dict[str, float]] = {
    "bank_timeout": {
        "RETRY": 0.72,
        "PAYMENT_LINK": 0.48,
        "REMINDER": 0.32,
        "ESCALATE": 0.45,
        "STOP": 0.0,
    },
    "network_error": {
        "RETRY": 0.76,
        "PAYMENT_LINK": 0.46,
        "REMINDER": 0.28,
        "ESCALATE": 0.42,
        "STOP": 0.0,
    },
    "insufficient_funds": {
        "RETRY": 0.16,
        "PAYMENT_LINK": 0.62,
        "REMINDER": 0.52,
        "ESCALATE": 0.38,
        "STOP": 0.0,
    },
    "authentication_failed": {
        "RETRY": 0.10,
        "PAYMENT_LINK": 0.70,
        "REMINDER": 0.46,
        "ESCALATE": 0.30,
        "STOP": 0.0,
    },
    "payment_method_issue": {
        "RETRY": 0.08,
        "PAYMENT_LINK": 0.66,
        "REMINDER": 0.40,
        "ESCALATE": 0.35,
        "STOP": 0.0,
    },
    "technical_error": {
        "RETRY": 0.58,
        "PAYMENT_LINK": 0.44,
        "REMINDER": 0.26,
        "ESCALATE": 0.46,
        "STOP": 0.0,
    },
    "unknown": {
        "RETRY": 0.32,
        "PAYMENT_LINK": 0.42,
        "REMINDER": 0.30,
        "ESCALATE": 0.36,
        "STOP": 0.0,
    },
}


@dataclass
class RecoveryRecord:
    """Represents a single recovery action event and its outcome."""

    # Identification
    record_id: str
    customer_id: str
    payment_id: str

    # Customer context
    customer_payment_count: int
    customer_success_rate: float
    customer_previous_failures: int

    # Payment context
    payment_amount: float
    currency: str
    payment_method: str
    failure_reason: str
    attempt_number: int
    hours_since_failure: float

    # Recovery context
    action_taken: str
    previous_action: str | None
    previous_attempt_count: int

    # Outcome
    recovered: bool
    amount_recovered: float
    recovery_time_hours: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _generate_customer_profile(rng: random.Random) -> dict[str, Any]:
    """Generate a realistic customer payment history profile."""
    # Most customers have 2 to 30 transactions
    payment_count = max(1, int(rng.expovariate(1 / 12) + 1))
    payment_count = min(payment_count, 100)

    # Customer historical success rates cluster around 75-95% with occasional lower outliers
    is_high_tier = rng.random() < 0.80
    if is_high_tier:
        success_rate = round(rng.uniform(0.70, 0.98), 3)
    else:
        success_rate = round(rng.uniform(0.25, 0.69), 3)

    previous_failures = round(payment_count * (1.0 - success_rate))

    return {
        "customer_id": f"cust_{uuid.UUID(int=rng.getrandbits(128)).hex[:12]}",
        "customer_payment_count": payment_count,
        "customer_success_rate": success_rate,
        "customer_previous_failures": previous_failures,
    }


def _select_action(
    failure_reason: str,
    attempt_number: int,
    previous_action: str | None,
    rng: random.Random,
) -> str:
    """Select a realistic recovery action based on failure type and attempt count."""
    if attempt_number > 4:
        # After repeated failures, stop or escalate
        return rng.choices(["STOP", "ESCALATE"], weights=[0.80, 0.20])[0]

    if attempt_number == 1:
        if failure_reason in ["bank_timeout", "network_error", "technical_error"]:
            return rng.choices(
                ["RETRY", "PAYMENT_LINK", "REMINDER", "ESCALATE"],
                weights=[0.75, 0.15, 0.08, 0.02],
            )[0]
        elif failure_reason in ["insufficient_funds"]:
            return rng.choices(
                ["PAYMENT_LINK", "REMINDER", "RETRY", "ESCALATE"],
                weights=[0.45, 0.35, 0.18, 0.02],
            )[0]
        elif failure_reason in ["authentication_failed", "payment_method_issue"]:
            return rng.choices(
                ["PAYMENT_LINK", "REMINDER", "RETRY", "ESCALATE"],
                weights=[0.60, 0.25, 0.12, 0.03],
            )[0]
        else:
            return rng.choices(
                ["RETRY", "PAYMENT_LINK", "REMINDER", "ESCALATE"],
                weights=[0.40, 0.35, 0.20, 0.05],
            )[0]
    else:
        # Subsequent attempts
        if attempt_number == 2:
            if previous_action == "RETRY":
                return rng.choices(
                    ["PAYMENT_LINK", "REMINDER", "RETRY", "STOP"],
                    weights=[0.45, 0.30, 0.20, 0.05],
                )[0]
            else:
                return rng.choices(
                    ["PAYMENT_LINK", "REMINDER", "ESCALATE", "STOP"],
                    weights=[0.40, 0.35, 0.15, 0.10],
                )[0]
        elif attempt_number == 3:
            return rng.choices(
                ["PAYMENT_LINK", "REMINDER", "ESCALATE", "STOP"],
                weights=[0.30, 0.30, 0.25, 0.15],
            )[0]
        else:
            return rng.choices(
                ["ESCALATE", "STOP", "REMINDER"],
                weights=[0.45, 0.40, 0.15],
            )[0]


def _calculate_recovery_probability(
    failure_reason: str,
    action_taken: str,
    attempt_number: int,
    customer_success_rate: float,
    customer_previous_failures: int,
    hours_since_failure: float,
    payment_amount: float,
) -> float:
    """
    Calculate the realistic recovery probability incorporating:
    - Baseline action effectiveness by failure reason
    - Customer history modifier (success rate & failure history)
    - Attempt decay
    - Time-lapse effects
    """
    if action_taken == "STOP":
        return 0.0

    base_prob = BASE_RECOVERY_PROBABILITIES.get(failure_reason, {}).get(
        action_taken, 0.30
    )

    # 1. Attempt degradation: Each subsequent attempt has lower marginal return
    attempt_decay = 0.88 ** (attempt_number - 1)
    prob = base_prob * attempt_decay

    # 2. Customer historical quality modifier (+/- 15%)
    customer_factor = (customer_success_rate - 0.75) * 0.20
    failure_penalty = min(customer_previous_failures * 0.015, 0.12)
    prob += customer_factor - failure_penalty

    # 3. Timing effect
    if action_taken == "RETRY":
        # Retries are most effective within the first 6 hours
        if hours_since_failure > 12.0:
            prob *= 0.80
    elif (
        action_taken in ["PAYMENT_LINK", "REMINDER"]
        and failure_reason == "insufficient_funds"
        and hours_since_failure >= 4.0
    ):
        # Insufficient funds recover slightly better after a few hours delay
        prob *= 1.10

    # 4. Very high amounts have slightly lower recovery resistance
    if payment_amount > 20000.0:
        prob *= 0.92

    # Clamp probability to reasonable probabilistic bounds [0.03, 0.95]
    return max(0.03, min(prob, 0.95))


def generate_synthetic_recovery_dataset(
    num_records: int = 5000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate a deterministic synthetic historical dataset of recovery records.

    Parameters:
    - num_records: Target total number of historical recovery records.
    - seed: Random seed for reproducibility.

    Returns:
    - List of dictionary records matching the RecoveryRecord schema.
    """
    rng = random.Random(seed)

    # Pre-generate a customer pool to simulate recurring customer interactions
    num_customers = max(200, num_records // 3)
    customer_pool = [_generate_customer_profile(rng) for _ in range(num_customers)]

    records: list[dict[str, Any]] = []

    # Failure reasons distribution weights
    failure_weights = [0.26, 0.22, 0.18, 0.14, 0.08, 0.08, 0.04]

    # Payment methods distribution weights
    payment_method_weights = [0.38, 0.42, 0.14, 0.06]

    payment_seq = 1

    while len(records) < num_records:
        customer = rng.choice(customer_pool)
        payment_id = f"pay_{uuid.UUID(int=rng.getrandbits(128)).hex[:14]}"
        payment_seq += 1

        # Payment amount: blend of common retail (< 3000), mid-tier, and high-value
        amount_tier = rng.random()
        if amount_tier < 0.60:
            amount = round(rng.uniform(199.0, 2999.0), 2)
        elif amount_tier < 0.90:
            amount = round(rng.uniform(3000.0, 12000.0), 2)
        else:
            amount = round(rng.uniform(12001.0, 49999.0), 2)

        payment_method = rng.choices(PAYMENT_METHODS, weights=payment_method_weights)[0]
        failure_reason = rng.choices(FAILURE_REASONS, weights=failure_weights)[0]

        # Simulate recovery attempts lifecycle for this payment
        attempt_number = 1
        previous_action: str | None = None
        current_hours = round(rng.uniform(0.1, 2.0), 2)
        payment_recovered = False

        max_attempts_for_payment = rng.choices(
            [1, 2, 3, 4], weights=[0.40, 0.35, 0.18, 0.07]
        )[0]

        while attempt_number <= max_attempts_for_payment and not payment_recovered:
            action_taken = _select_action(
                failure_reason=failure_reason,
                attempt_number=attempt_number,
                previous_action=previous_action,
                rng=rng,
            )

            recovery_prob = _calculate_recovery_probability(
                failure_reason=failure_reason,
                action_taken=action_taken,
                attempt_number=attempt_number,
                customer_success_rate=customer["customer_success_rate"],
                customer_previous_failures=customer["customer_previous_failures"],
                hours_since_failure=current_hours,
                payment_amount=amount,
            )

            recovered = rng.random() < recovery_prob

            if recovered:
                amount_recovered = amount
                recovery_time_hours = round(current_hours + rng.uniform(0.05, 2.5), 2)
                payment_recovered = True
            else:
                amount_recovered = 0.0
                recovery_time_hours = 0.0

            record = RecoveryRecord(
                record_id=f"rec_{uuid.UUID(int=rng.getrandbits(128)).hex[:14]}",
                customer_id=customer["customer_id"],
                payment_id=payment_id,
                customer_payment_count=customer["customer_payment_count"],
                customer_success_rate=customer["customer_success_rate"],
                customer_previous_failures=customer["customer_previous_failures"],
                payment_amount=amount,
                currency="INR",
                payment_method=payment_method,
                failure_reason=failure_reason,
                attempt_number=attempt_number,
                hours_since_failure=current_hours,
                action_taken=action_taken,
                previous_action=previous_action,
                previous_attempt_count=attempt_number - 1,
                recovered=recovered,
                amount_recovered=amount_recovered,
                recovery_time_hours=recovery_time_hours,
            )

            records.append(record.to_dict())

            if len(records) == num_records:
                break

            if action_taken == "STOP":
                break

            # Advance to next attempt
            previous_action = action_taken
            attempt_number += 1
            current_hours = round(current_hours + rng.uniform(2.0, 18.0), 2)

    return records


def save_dataset_to_csv(records: list[dict[str, Any]], filepath: str | Path) -> None:
    """Save record list to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = list(records[0].keys())
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_dataset_from_csv(filepath: str | Path) -> list[dict[str, Any]]:
    """Load dataset records from a CSV file with typed conversion."""
    path = Path(filepath)
    records: list[dict[str, Any]] = []

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "record_id": row["record_id"],
                    "customer_id": row["customer_id"],
                    "payment_id": row["payment_id"],
                    "customer_payment_count": int(row["customer_payment_count"]),
                    "customer_success_rate": float(row["customer_success_rate"]),
                    "customer_previous_failures": int(
                        row["customer_previous_failures"]
                    ),
                    "payment_amount": float(row["payment_amount"]),
                    "currency": row["currency"],
                    "payment_method": row["payment_method"],
                    "failure_reason": row["failure_reason"],
                    "attempt_number": int(row["attempt_number"]),
                    "hours_since_failure": float(row["hours_since_failure"]),
                    "action_taken": row["action_taken"],
                    "previous_action": row["previous_action"]
                    if row["previous_action"]
                    else None,
                    "previous_attempt_count": int(row["previous_attempt_count"]),
                    "recovered": row["recovered"].lower() in ("true", "1"),
                    "amount_recovered": float(row["amount_recovered"]),
                    "recovery_time_hours": float(row["recovery_time_hours"]),
                }
            )

    return records


def calculate_dataset_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate comprehensive statistics and recovery rates across segments."""
    total_records = len(records)
    if total_records == 0:
        return {}

    unique_customers = len({r["customer_id"] for r in records})
    unique_payments = len({r["payment_id"] for r in records})
    unique_payment_amounts = {r["payment_id"]: r["payment_amount"] for r in records}
    total_amount_at_risk = round(sum(unique_payment_amounts.values()), 2)
    total_amount_recovered = round(sum(r["amount_recovered"] for r in records), 2)
    recovered_records = sum(1 for r in records if r["recovered"])
    overall_recovery_rate = round(recovered_records / total_records, 4)

    # Recovery rate by failure reason
    by_reason: dict[str, dict[str, Any]] = {}
    for reason in FAILURE_REASONS:
        subset = [r for r in records if r["failure_reason"] == reason]
        count = len(subset)
        recovered = sum(1 for r in subset if r["recovered"])
        rate = round(recovered / count, 4) if count > 0 else 0.0
        by_reason[reason] = {
            "attempts": count,
            "recovered": recovered,
            "recovery_rate": rate,
        }

    # Recovery rate by action
    by_action: dict[str, dict[str, Any]] = {}
    for action in RECOVERY_ACTIONS:
        subset = [r for r in records if r["action_taken"] == action]
        count = len(subset)
        recovered = sum(1 for r in subset if r["recovered"])
        rate = round(recovered / count, 4) if count > 0 else 0.0
        by_action[action] = {
            "attempts": count,
            "recovered": recovered,
            "recovery_rate": rate,
        }

    # Recovery rate by attempt number
    by_attempt: dict[int, dict[str, Any]] = {}
    attempts_seen = sorted({r["attempt_number"] for r in records})
    for att in attempts_seen:
        subset = [r for r in records if r["attempt_number"] == att]
        count = len(subset)
        recovered = sum(1 for r in subset if r["recovered"])
        rate = round(recovered / count, 4) if count > 0 else 0.0
        by_attempt[att] = {
            "attempts": count,
            "recovered": recovered,
            "recovery_rate": rate,
        }

    # Recovery rate by (failure_reason, action) cross-tab
    cross_tab: dict[str, dict[str, float]] = {}
    for reason in FAILURE_REASONS:
        cross_tab[reason] = {}
        for action in RECOVERY_ACTIONS:
            subset = [
                r
                for r in records
                if r["failure_reason"] == reason and r["action_taken"] == action
            ]
            count = len(subset)
            recovered = sum(1 for r in subset if r["recovered"])
            cross_tab[reason][action] = (
                round(recovered / count, 4) if count > 0 else 0.0
            )

    return {
        "total_records": total_records,
        "unique_customers": unique_customers,
        "unique_payments": unique_payments,
        "total_amount_at_risk": total_amount_at_risk,
        "total_amount_recovered": total_amount_recovered,
        "overall_recovery_rate": overall_recovery_rate,
        "recovery_rate_by_failure_reason": by_reason,
        "recovery_rate_by_action": by_action,
        "recovery_rate_by_attempt_number": by_attempt,
        "cross_tab_reason_action": cross_tab,
    }


def main():
    """Generate default dataset artifact and print statistics."""
    print(
        "Generating Revora synthetic historical recovery dataset (5,000 records, seed=42)..."
    )
    records = generate_synthetic_recovery_dataset(num_records=5000, seed=42)

    # Save to data directory
    output_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "historical_recovery_data.csv"
    )
    save_dataset_to_csv(records, output_path)
    print(f"Dataset successfully saved to: {output_path}")

    stats = calculate_dataset_statistics(records)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total Records:           {stats['total_records']}")
    print(f"Unique Customers:        {stats['unique_customers']}")
    print(f"Unique Payments:         {stats['unique_payments']}")
    print(f"Total Amount at Risk:    INR {stats['total_amount_at_risk']:,.2f}")
    print(f"Total Amount Recovered:  INR {stats['total_amount_recovered']:,.2f}")
    print(f"Overall Recovery Rate:   {stats['overall_recovery_rate'] * 100:.2f}%")

    print("\n--- Recovery Rate by Failure Reason ---")
    for reason, data in stats["recovery_rate_by_failure_reason"].items():
        print(
            f"  {reason:<25} Attempts: {data['attempts']:<5} Recovered: {data['recovered']:<5} Rate: {data['recovery_rate'] * 100:.2f}%"
        )

    print("\n--- Recovery Rate by Action Taken ---")
    for action, data in stats["recovery_rate_by_action"].items():
        print(
            f"  {action:<15} Attempts: {data['attempts']:<5} Recovered: {data['recovered']:<5} Rate: {data['recovery_rate'] * 100:.2f}%"
        )

    print("\n--- Recovery Rate by Attempt Number ---")
    for att, data in stats["recovery_rate_by_attempt_number"].items():
        print(
            f"  Attempt {att:<5} Attempts: {data['attempts']:<5} Recovered: {data['recovered']:<5} Rate: {data['recovery_rate'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
