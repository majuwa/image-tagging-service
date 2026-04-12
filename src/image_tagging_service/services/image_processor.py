import io

import structlog
from PIL import Image

logger = structlog.get_logger()


class ImageProcessor:
    def __init__(self, max_dimension: int = 1024):
        self.max_dimension = max_dimension

    def scale_image(self, image_data: bytes, max_dimension: int | None = None) -> Image.Image:
        """Scale image to fit within max_dimension while preserving aspect ratio."""
        max_dim = max_dimension or self.max_dimension
        image = Image.open(io.BytesIO(image_data))

        # Convert RGBA/P to RGB
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        width, height = image.size
        if max(width, height) > max_dim:
            ratio = max_dim / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info("image_scaled", original=(width, height), scaled=new_size)

        return image

    def validate_format(self, content_type: str, supported: list[str]) -> bool:
        """Check whether the given content type is in the supported list."""
        return content_type in supported
