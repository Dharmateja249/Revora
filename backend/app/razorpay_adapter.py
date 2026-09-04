"""
Revora Razorpay Gateway Adapter.

Encapsulates external communication with the Razorpay Payment Gateway.
Provides safe, isolated, and dry-run capable payment link generation and
recovery action execution without leaking secrets or credentials.
"""

import logging
from decimal import ROUND_HALF_UP, Decimal
from typing import Any
from uuid import uuid4

import httpx

from app.config import get_settings

logger = logging.getLogger("revora.razorpay_adapter")


class RazorpayError(Exception):
    """Base exception for all Razorpay adapter errors."""


class RazorpayConfigurationError(RazorpayError):
    """Raised when Razorpay credentials or configuration are invalid."""


class RazorpayAPIError(RazorpayError):
    """Raised when an external Razorpay API call fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class RazorpayAdapter:
    """
    Adapter client for Razorpay APIs.

    Supports both live HTTP Basic Auth execution and safe dry-run/simulation mode
    for local demos, integration testing, and CI environments without real financial transactions.
    """

    def __init__(
        self,
        key_id: str | None = None,
        key_secret: str | None = None,
        base_url: str | None = None,
        dry_run: bool | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        settings = get_settings()

        self._key_id = (
            key_id if key_id is not None else getattr(settings, "RAZORPAY_KEY_ID", None)
        )
        self._key_secret = (
            key_secret
            if key_secret is not None
            else getattr(settings, "RAZORPAY_KEY_SECRET", None)
        )
        self._base_url = (
            base_url
            or getattr(settings, "RAZORPAY_BASE_URL", None)
            or "https://api.razorpay.com/v1"
        ).rstrip("/")

        resolved_dry_run = (
            dry_run
            if dry_run is not None
            else getattr(settings, "RAZORPAY_DRY_RUN", True)
        )
        self._dry_run = bool(resolved_dry_run)
        self._client = client

        if not self._dry_run and not self._base_url.lower().startswith("https://"):
            raise RazorpayConfigurationError(
                "Live Razorpay API requests require a secure HTTPS base URL."
            )

    @property
    def dry_run(self) -> bool:
        """True if adapter is operating in safe dry-run / simulation mode."""
        return self._dry_run

    @property
    def base_url(self) -> str:
        """Base URL for Razorpay API requests."""
        return self._base_url

    def __repr__(self) -> str:
        masked_secret = "***" if self._key_secret else "None"
        return (
            f"RazorpayAdapter(key_id={self._key_id!r}, key_secret={masked_secret}, "
            f"base_url={self._base_url!r}, dry_run={self._dry_run})"
        )

    async def create_payment_link(
        self,
        amount: float,
        currency: str = "INR",
        description: str = "Payment recovery link",
        customer_name: str | None = None,
        customer_email: str | None = None,
        customer_contact: str | None = None,
        reference_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a Razorpay Payment Link (POST /v1/payment_links/).

        In Razorpay, amounts for INR must be denominated in the smallest currency unit (paise).

        Args:
            amount: Amount in major currency units (e.g. 100.0 for ₹100.00).
            currency: Three-letter currency code (e.g. 'INR').
            description: Customer-facing explanation of the payment link.
            customer_name: Optional customer full name.
            customer_email: Optional customer email address for notification.
            customer_contact: Optional customer phone number for SMS notification.
            reference_id: Optional internal reference ID.

        Returns:
            Dictionary containing payment link resource data (id, short_url, status, etc.).
        """
        if amount <= 0:
            raise ValueError(f"Amount must be positive, got {amount}")

        # Razorpay expects amounts in paise for INR; use Decimal conversion with ROUND_HALF_UP
        amount_in_subunits = int(
            (Decimal(str(amount)) * Decimal(100)).quantize(
                Decimal(1), rounding=ROUND_HALF_UP
            )
        )

        if self._dry_run:
            simulated_id = f"plink_sim_{uuid4().hex[:14]}"
            simulated_url = f"https://rzp.io/i/sim_{uuid4().hex[:8]}"
            logger.info(
                "Simulated Razorpay Payment Link creation: id=%s amount=%s %s reference_id=%s",
                simulated_id,
                amount,
                currency,
                reference_id,
            )
            return {
                "id": simulated_id,
                "short_url": simulated_url,
                "reference_id": reference_id,
                "status": "created",
                "amount": amount_in_subunits,
                "currency": currency.upper(),
                "description": description,
                "customer": {
                    "name": customer_name,
                    "email": customer_email,
                    "contact": customer_contact,
                },
                "simulated": True,
            }

        # Live Execution
        if not self._base_url.lower().startswith("https://"):
            raise RazorpayConfigurationError(
                "Live Razorpay API requests require a secure HTTPS base URL."
            )

        if not self._key_id or not self._key_secret:
            raise RazorpayConfigurationError(
                "Razorpay API credentials ('RAZORPAY_KEY_ID' and 'RAZORPAY_KEY_SECRET') "
                "must be configured for live payment link creation."
            )

        endpoint = f"{self._base_url}/payment_links/"
        payload: dict[str, Any] = {
            "amount": amount_in_subunits,
            "currency": currency.upper(),
            "description": description,
            "notify": {
                "sms": bool(customer_contact),
                "email": bool(customer_email),
            },
        }

        customer_obj: dict[str, str] = {}
        if customer_name:
            customer_obj["name"] = customer_name
        if customer_email:
            customer_obj["email"] = customer_email
        if customer_contact:
            customer_obj["contact"] = customer_contact
        if customer_obj:
            payload["customer"] = customer_obj

        if reference_id:
            payload["reference_id"] = reference_id

        auth = (self._key_id, self._key_secret)

        try:
            if self._client is not None:
                response = await self._client.post(
                    endpoint,
                    json=payload,
                    auth=auth,
                    timeout=15.0,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        endpoint,
                        json=payload,
                        auth=auth,
                        timeout=15.0,
                    )

            if response.status_code >= 400:
                logger.error(
                    "Razorpay API returned status %s for payment link creation.",
                    response.status_code,
                )
                error_msg = f"Razorpay API returned HTTP {response.status_code}"
                try:
                    err_json = response.json()
                    if isinstance(err_json, dict) and "error" in err_json:
                        sub_desc = err_json["error"].get("description")
                        if sub_desc:
                            error_msg = f"{error_msg}: {sub_desc}"
                except Exception:
                    logger.debug(
                        "Failed to parse Razorpay error JSON response", exc_info=True
                    )
                raise RazorpayAPIError(error_msg, status_code=response.status_code)

            try:
                data = response.json()
            except ValueError as exc:
                logger.error(
                    "Failed to parse JSON response from Razorpay API with status %s",
                    response.status_code,
                )
                raise RazorpayAPIError(
                    f"Invalid JSON payload returned from Razorpay API (status {response.status_code})",
                    status_code=response.status_code,
                ) from exc

            if not isinstance(data, dict):
                raise RazorpayAPIError(
                    f"Unexpected response structure from Razorpay API: expected dict, got {type(data).__name__}",
                    status_code=response.status_code,
                )

            return {
                "id": data.get("id"),
                "short_url": data.get("short_url"),
                "status": data.get("status", "created"),
                "amount": data.get("amount", amount_in_subunits),
                "currency": data.get("currency", currency.upper()),
                "description": data.get("description", description),
                "simulated": False,
            }

        except httpx.TimeoutException as exc:
            logger.warning("Timeout connecting to Razorpay Payment Links API: %s", exc)
            raise RazorpayAPIError("Timeout communicating with Razorpay API") from exc
        except httpx.RequestError as exc:
            logger.warning("Network error communicating with Razorpay API: %s", exc)
            raise RazorpayAPIError(
                "Network connection error while calling Razorpay API"
            ) from exc
