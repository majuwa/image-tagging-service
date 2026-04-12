from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from .api.dependencies import get_classifier, get_settings
from .api.middleware.auth import ApiKeyMiddleware
from .api.routes import classify, health
from .config import Settings
from .core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    logger = structlog.get_logger()
    logger.info("starting_service", model=settings.model_name)

    classifier = get_classifier()
    try:
        classifier.load_model()
    except Exception as exc:  # noqa: BLE001
        classifier._load_error = str(exc)  # noqa: SLF001
        logger.warning(
            "model_load_failed",
            model=settings.model_name,
            error=str(exc),
            hint=(
                f"Could not load '{settings.model_name}'. "
                "Check model name, available disk space, and internet connection. "
                "Model page: https://huggingface.co/google/gemma-4-E2B-it"
            ),
        )
        logger.warning(
            "service_degraded",
            detail="classify endpoint will return 503 until model is available",
        )

    yield

    logger.info("shutting_down")


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="Image Tagging Service",
        description=(
            "AI-powered image classification and hierarchical tag suggestion service using Gemma 4"
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    if settings.auth_enabled and settings.api_key:
        app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)

    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(classify.router, prefix="/api/v1", tags=["classification"])

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "image_tagging_service.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
