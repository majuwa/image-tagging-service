from pydantic import BaseModel, Field


class HierarchicalTag(BaseModel):
    """A tag with its full hierarchical path."""

    path: list[str] = Field(
        ..., description="Tag path from root to leaf, e.g. ['Nature', 'Animals', 'Birds']"
    )


class ClassifyRequest(BaseModel):
    """Metadata for classification request (sent as form field alongside image)."""

    existing_tags: list[HierarchicalTag] = Field(
        default_factory=list, description="Existing hierarchical tags in the catalog"
    )
    max_new_tags: int = Field(default=10, ge=1, le=50)
    confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
