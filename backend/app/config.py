import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    APP_NAME: str = "Revora"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    DATABASE_URL: str = "sqlite:///./revora.db"
    ENABLE_AGENT_DECISION_ENGINE: bool = False

    # Authentication & Security
    AUTH_SECRET_KEY: str = "revora-demo-auth-secret-key-32bytesmin!"
    ENABLE_DEMO_AUTH_ENDPOINT: bool = True
    DEMO_CUSTOMER_IDS: tuple[str, ...] = ("e9cd4c97-979b-4753-9925-640623f74eee",)

    # LLM Provider Configuration
    LLM_PROVIDER: str = "mock"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str | None = None
    OPENAI_TIMEOUT_SECONDS: float | None = None
    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str | None = None
    GEMINI_TIMEOUT_SECONDS: float | None = None
    HF_TOKEN: str | None = None
    HF_MODEL: str | None = "Qwen/Qwen3-32B"
    HF_TIMEOUT_SECONDS: float | None = 30.0
    LLM_API_KEY: str | None = None
    LLM_MODEL: str | None = None
    LLM_TIMEOUT_SECONDS: float | None = None

    # Razorpay Gateway Configuration
    RAZORPAY_KEY_ID: str | None = None
    RAZORPAY_KEY_SECRET: str | None = None
    RAZORPAY_BASE_URL: str = "https://api.razorpay.com/v1"
    RAZORPAY_DRY_RUN: bool = True

    model_config = {
        "env_file": None if os.environ.get("REVORA_TESTING") == "1" else ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def model_post_init(self, context: object, /) -> None:
        """Validate that live Razorpay API calls strictly require an HTTPS endpoint."""
        super().model_post_init(context)
        if not self.RAZORPAY_DRY_RUN and not self.RAZORPAY_BASE_URL.lower().startswith(
            "https://"
        ):
            raise ValueError(
                "Live Razorpay API requests require a secure HTTPS base URL."
            )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
