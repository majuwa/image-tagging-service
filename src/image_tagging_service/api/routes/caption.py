import json
import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import get_classifier, get_image_processor, get_settings
from ..models.responses import CaptionResponse

router = APIRouter()
logger = structlog.get_logger()


@router.post("/caption", response_model=CaptionResponse)
async def caption_photo(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    tags: Annotated[str | None, Form(description="JSON array of keyword name strings")] = None,
    location: Annotated[str | None, Form(description="GPS string, e.g. '48.8566, 2.3522'")] = None,
    city: Annotated[str | None, Form()] = None,
    country: Annotated[str | None, Form()] = None,
    date_taken: Annotated[str | None, Form(description="Formatted capture date/time")] = None,
    camera: Annotated[str | None, Form(description="Camera model name")] = None,
    language: Annotated[str | None, Form(description="Caption language ISO 639-1 code, e.g. 'de', 'en'")] = None,
    classifier=Depends(get_classifier),
    image_processor=Depends(get_image_processor),
    settings=Depends(get_settings),
) -> CaptionResponse:
    start_time = time.monotonic()
    logger.info(
        "caption_request",
        filename=image.filename,
        has_tags=tags is not None,
        city=city,
        country=country,
    )

    if image.content_type not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {image.content_type}",
        )

    image_data = await image.read()
    if len(image_data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.max_upload_size_mb}MB limit",
        )

    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{settings.model_name}' is not loaded. "
                f"Reason: {classifier._load_error or 'unknown'}."  # noqa: SLF001
            ),
        )

    # Parse optional context
    tag_list: list[str] = []
    if tags:
        try:
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                tag_list = [str(t) for t in parsed]
        except (json.JSONDecodeError, ValueError):
            pass

    context = {
        "tags": tag_list or None,
        "location": location,
        "city": city,
        "country": country,
        "date_taken": date_taken,
        "camera": camera,
        "language": language or "de",
    }
    # Remove None values so the classifier can test `if ctx.get(key)` cleanly
    context = {k: v for k, v in context.items() if v is not None}

    scaled_image = image_processor.scale_image(image_data)
    result = classifier.caption(scaled_image, context=context or None)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    logger.info("caption_response", elapsed_ms=elapsed_ms)

    return CaptionResponse(
        caption=result["caption"],
        hashtags=result["hashtags"],
        processing_time_ms=elapsed_ms,
        model_name=settings.model_name,
    )
