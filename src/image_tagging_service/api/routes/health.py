from fastapi import APIRouter, Depends

from ..dependencies import get_classifier, get_settings
from ..models.responses import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    classifier=Depends(get_classifier),
) -> HealthResponse:
    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        model_loaded=classifier.is_loaded,
        model_name=classifier.model_name,
        model_error=classifier._load_error,  # noqa: SLF001
        version="0.1.0",
    )


@router.get("/models/info", response_model=ModelInfoResponse)
async def model_info(
    classifier=Depends(get_classifier),
    settings=Depends(get_settings),
) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=classifier.model_name,
        device=str(settings.device),
        max_image_dimension=settings.max_image_dimension,
        supported_formats=settings.supported_formats,
    )
