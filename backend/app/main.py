import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings
from app.database import init_db
from app.observability import reset_request_id, sanitize_request_id, set_request_id
from app.vector_seeder import seed_runtime_vector_index

logger = logging.getLogger("revora.api")
settings = get_settings()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    HTTP middleware establishing request-level correlation context via ContextVar.
    Preserves incoming valid X-Request-ID, generates one when absent/invalid, logs lifecycle,
    injects X-Request-ID into response headers, and guarantees cleanup in finally.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        raw_header = request.headers.get("X-Request-ID")
        request_id = sanitize_request_id(raw_header)
        token = set_request_id(request_id)

        start_time = time.perf_counter()
        logger.info(
            "HTTP request started: method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "HTTP request completed: method=%s path=%s status=%d duration_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                request_id,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.exception(
                "HTTP request failed with unhandled exception: method=%s path=%s duration_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                duration_ms,
                request_id,
            )
            raise
        finally:
            reset_request_id(token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on application startup
    init_db()
    # Seed runtime vector index with historical recovery precedents for demo tenant
    seed_runtime_vector_index()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Enable request context & correlation tracking middleware
app.add_middleware(RequestContextMiddleware)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from app.routers.dashboard import router as dashboard_router
from app.routers.decision import auth_router
from app.routers.decision import router as decision_router
from app.routers.recovery import router as recovery_router

# Register routers
app.include_router(auth_router)
app.include_router(decision_router)
app.include_router(dashboard_router)
app.include_router(
    recovery_router,
    prefix="/v1/recovery",
    tags=["Recovery"],
)


@app.get("/health", tags=["Health"])
def health_check():
    """Simple health check endpoint returning service status and runtime gateway mode."""
    gateway_mode = "dry_run" if settings.RAZORPAY_DRY_RUN else "live"
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "gateway_mode": gateway_mode,
    }
