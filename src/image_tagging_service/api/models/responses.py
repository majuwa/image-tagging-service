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


class RatingResponse(BaseModel):
    """Response from the image rating endpoint."""

    rating: int = Field(..., ge=1, le=5, description="AI-suggested star rating (1–5)")
    reasoning: str = Field(..., description="Short explanation for the rating")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_name: str = Field(..., description="Model used for rating")


class ReviewResponse(BaseModel):
    """Response from the AI photo review endpoint."""

    composition: str = Field(..., description="Analysis of image composition")
    image_quality: str = Field(..., description="Technical quality assessment")
    subject: str = Field(..., description="Subject and how it is captured")
    editing_tips: str = Field(..., description="Specific editing recommendations")
    mood: str = Field(..., description="Mood and atmosphere of the image")
    overall: str = Field(..., description="Overall summary and key recommendations")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_name: str = Field(..., description="Model used for review")



class CaptionResponse(BaseModel):
    """Response from the AI social media caption endpoint."""

    caption: str = Field(..., description="Generated Instagram-style caption")
    hashtags: str = Field(..., description="Space-separated hashtags (each prefixed with #)")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    model_name: str = Field(..., description="Model used for caption generation")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    device: str
    max_image_dimension: int
    supported_formats: list[str]
