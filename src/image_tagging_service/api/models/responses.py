from pydantic import BaseModel, Field


class TagSuggestion(BaseModel):
    """A single tag suggestion with confidence score."""

    path: list[str] = Field(..., description="Hierarchical tag path")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_new: bool = Field(..., description="Whether this is a new tag not in existing taxonomy")


class ClassifyResponse(BaseModel):
    """Response from the classification endpoint."""

    matched_tags: list[TagSuggestion] = Field(default_factory=list)
    new_tags: list[TagSuggestion] = Field(default_factory=list)
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_name: str = Field(..., description="Model used for classification")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy" or "degraded"
    model_loaded: bool
    model_name: str | None
    model_error: str | None = Field(None, description="Load error message when model is not loaded")
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    device: str
    max_image_dimension: int
    supported_formats: list[str]
