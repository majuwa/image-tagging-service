import json
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..dependencies import get_classifier, get_image_processor, get_settings, get_tag_matcher
from ..models.requests import HierarchicalTag
from ..models.responses import ClassifyResponse, TagSuggestion

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_image(
    image: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    existing_tags: str = Form(default="[]", description="JSON array of hierarchical tags"),
    max_new_tags: int = Form(default=10, ge=1, le=50),
    confidence_threshold: float = Form(default=0.3, ge=0.0, le=1.0),
    classifier=Depends(get_classifier),
    image_processor=Depends(get_image_processor),
    tag_matcher=Depends(get_tag_matcher),
    settings=Depends(get_settings),
) -> ClassifyResponse:
    start_time = time.monotonic()

    # Validate content type
    if image.content_type not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {image.content_type}",
        )

    # Validate file size
    image_data = await image.read()
    if len(image_data) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.max_upload_size_mb}MB limit",
        )

    # Parse existing tags
    try:
        tags_raw = json.loads(existing_tags)
        parsed_tags = [
            HierarchicalTag(**t) if isinstance(t, dict) else HierarchicalTag(path=t)
            for t in tags_raw
        ]
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid existing_tags JSON: {e}") from e

    # Scale image
    scaled_image = image_processor.scale_image(image_data)

    # Classify
    existing_dicts = [{"path": t.path} for t in parsed_tags]
    suggestions = classifier.classify(scaled_image, existing_dicts, max_new_tags)

    # Match against existing
    flat_existing = tag_matcher.flatten_tag_paths(existing_dicts)
    matched, new = tag_matcher.match_tags(suggestions, flat_existing, confidence_threshold)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    return ClassifyResponse(
        matched_tags=[TagSuggestion(**m) for m in matched],
        new_tags=[TagSuggestion(**n) for n in new[:max_new_tags]],
        processing_time_ms=elapsed_ms,
        model_name=settings.model_name,
    )
