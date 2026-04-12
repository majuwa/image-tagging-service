import os

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ITS_", env_file=".env")

    # Server
    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    debug: bool = False

    # Model
    model_name: str = "google/gemma-4-4b-it"
    device: str = "auto"  # auto, cpu, cuda, mps
    # HuggingFace token for gated models (e.g. Gemma 4).
    # Also read from the standard HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars.
    hf_token: str | None = None

    @model_validator(mode="after")
    def _resolve_hf_token(self) -> "Settings":
        """Fall back to standard HuggingFace env vars when ITS_HF_TOKEN is unset."""
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
                "HUGGING_FACE_HUB_TOKEN"
            )
        return self

    # Image processing
    max_image_dimension: int = 1024
    supported_formats: list[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"],
    )
    max_upload_size_mb: int = 50

    # Classification
    default_confidence_threshold: float = 0.3
    default_max_new_tags: int = 10
    max_tags_in_request: int = 5000

    # Auth
    auth_enabled: bool = False
    api_key: str | None = None

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
