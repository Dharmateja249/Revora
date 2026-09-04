from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import get_settings

settings = get_settings()

# SQLite requires 'check_same_thread: False' to allow multi-threaded access in FastAPI
connect_args = (
    {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator:
    """Yield a database session and ensure it is closed after request lifecycle."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all database tables defined in the application models and ensure schema consistency."""
    import app.models  # noqa: F401 (Ensure models are imported and registered with Base)

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if "recovery_attempts" in inspector.get_table_names():
        columns = {col["name"] for col in inspector.get_columns("recovery_attempts")}
        if "idempotency_key" not in columns:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE recovery_attempts ADD COLUMN idempotency_key VARCHAR(255)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS ix_recovery_attempts_idempotency_key "
                        "ON recovery_attempts (idempotency_key)"
                    )
                )
