import time

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..dependencies import get_classifier, get_image_processor, get_settings
from ..models.responses import RatingResponse

router = APIRouter()
logger = structlog.get_logger()


@router.post("/rate", response_model=RatingResponse)
async def rate_image(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    classifier=Depends(get_classifier),
    image_processor=Depends(get_image_processor),
    settings=Depends(get_settings),
) -> RatingResponse:
    start_time = time.monotonic()
    logger.info(
        "rate_request",
        filename=image.filename,
        content_type=image.content_type,
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

    scaled_image = image_processor.scale_image(image_data)
    result = classifier.rate(scaled_image)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "rate_response",
        rating=result["rating"],
        elapsed_ms=elapsed_ms,
    )

    return RatingResponse(
        rating=result["rating"],
        reasoning=result["reasoning"],
        processing_time_ms=elapsed_ms,
        model_name=settings.model_name,
    )
