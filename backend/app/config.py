from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    APP_NAME: str = "Revora"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    DATABASE_URL: str = "sqlite:///./revora.db"
    ENABLE_AGENT_DECISION_ENGINE: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
