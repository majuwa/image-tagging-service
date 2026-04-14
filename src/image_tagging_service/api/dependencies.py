from functools import lru_cache

from ..config import Settings
from ..services.classifier import ImageClassifier
from ..services.image_processor import ImageProcessor
from ..services.tag_matcher import TagMatcher

_classifier: ImageClassifier | None = None
_image_processor: ImageProcessor | None = None
_tag_matcher: TagMatcher | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_classifier() -> ImageClassifier:
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        settings = get_settings()
        _classifier = ImageClassifier(
            model_name=settings.model_name,
            llm_base_url=settings.llm_base_url,
        )
    return _classifier


def get_image_processor() -> ImageProcessor:
    global _image_processor  # noqa: PLW0603
    if _image_processor is None:
        settings = get_settings()
        _image_processor = ImageProcessor(max_dimension=settings.max_image_dimension)
    return _image_processor


def get_tag_matcher() -> TagMatcher:
    global _tag_matcher  # noqa: PLW0603
    if _tag_matcher is None:
        _tag_matcher = TagMatcher()
    return _tag_matcher
