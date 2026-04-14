import time

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..dependencies import get_classifier, get_image_processor, get_settings
from ..models.responses import ReviewResponse

router = APIRouter()
logger = structlog.get_logger()


@router.post("/review", response_model=ReviewResponse)
async def review_image(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    classifier=Depends(get_classifier),
    image_processor=Depends(get_image_processor),
    settings=Depends(get_settings),
) -> ReviewResponse:
    start_time = time.monotonic()
    logger.info(
        "review_request",
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
    result = classifier.review(scaled_image)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    logger.info("review_response", elapsed_ms=elapsed_ms)

    return ReviewResponse(
        composition=result["composition"],
        image_quality=result["image_quality"],
        subject=result["subject"],
        editing_tips=result["editing_tips"],
        mood=result["mood"],
        overall=result["overall"],
        processing_time_ms=elapsed_ms,
        model_name=settings.model_name,
    )
