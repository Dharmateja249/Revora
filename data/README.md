# Revora Synthetic Historical Recovery Dataset

This dataset represents **Historical Recovery Memory** for Revora's intelligence layer. It captures historical payment failure events, customer risk context, recovery actions taken, and the resulting recovery outcomes.

---

## 1. How to Generate the Dataset

The authoritative source of truth is the generator module in [`backend/app/synthetic_data.py`](../backend/app/synthetic_data.py).

To reproduce or regenerate the dataset:

```bash
# From the backend directory with your virtual environment active:
python app/synthetic_data.py
```

This generates `data/historical_recovery_data.csv` with exactly 5,000 deterministic records using fixed random seed `42`.

---

## 2. Dataset Schema & Field Definitions

Each record represents a single recovery attempt/action lifecycle event:

| Field | Type | Description |
| :--- | :--- | :--- |
| `record_id` | `string` | Unique identifier for the recovery record event (e.g. `rec_...`). |
| `customer_id` | `string` | Identifier of the customer associated with the payment. |
| `payment_id` | `string` | Identifier of the failed payment transaction. |
| `customer_payment_count`| `integer`| Total historical payment transactions made by this customer. |
| `customer_success_rate` | `float` | Customer's historical payment success rate (0.0 to 1.0). |
| `customer_previous_failures` | `integer` | Number of previous payment failures on record for this customer. |
| `payment_amount` | `float` | Transaction value at risk in INR. |
| `currency` | `string` | Transaction currency (`INR`). |
| `payment_method` | `string` | Method used (`card`, `upi`, `netbanking`, `wallet`). |
| `failure_reason` | `string` | Controlled failure classification (`bank_timeout`, `insufficient_funds`, `authentication_failed`, `network_error`, `technical_error`, `payment_method_issue`, `unknown`). |
| `attempt_number` | `integer` | 1-indexed attempt number for this payment episode. |
| `hours_since_failure` | `float` | Elapsed time in hours between initial payment failure and this attempt. |
| `action_taken` | `string` | Recovery intervention chosen (`RETRY`, `PAYMENT_LINK`, `REMINDER`, `ESCALATE`, `STOP`). |
| `previous_action` | `string \| null` | Immediate prior recovery action taken, or null for attempt 1. |
| `previous_attempt_count` | `integer` | Count of attempts prior to this one (`attempt_number - 1`). |
| `recovered` | `boolean` | `True` if this action successfully recovered the payment; `False` otherwise. |
| `amount_recovered` | `float` | Total amount recovered (`payment_amount` if `recovered` is True, `0.0` if False). |
| `recovery_time_hours` | `float` | Total hours elapsed until recovery completion (0.0 if not recovered). |

---

## 3. Synthetic Behavior & Probabilistic Model

The generator simulates realistic payment recovery dynamics with probabilistic variation:

1. **Transient Network/Bank Failures**
   - Failures like `bank_timeout` and `network_error` respond well to automated `RETRY` (72–76% baseline success), especially in early attempts.
2. **Customer Intervention Needs**
   - Failures like `insufficient_funds`, `authentication_failed`, and `payment_method_issue` have configured baseline recovery probabilities of:
     - `RETRY`: 8–16%
     - `PAYMENT_LINK`: 62–70%
     - `REMINDER`: 40–52%
3. **Attempt Degradation Decay**
   - Each successive attempt has diminishing returns ($0.88^{\text{attempt} - 1}$).
4. **Customer Credit Profile**
   - Higher historical success rates positively bias recovery probability; high previous failure counts negatively bias recovery probability.
5. **Action Boundaries**
   - `STOP` always yields `recovered = False` and `amount_recovered = 0.0`.
   - `ESCALATE` has a low selection weight on early attempts and becomes more likely as recovery attempts increase.
