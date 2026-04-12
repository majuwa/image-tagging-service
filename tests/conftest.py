import io
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_tagging_service.api import dependencies
from image_tagging_service.config import Settings
from image_tagging_service.main import create_app
from image_tagging_service.services.classifier import ImageClassifier
from image_tagging_service.services.image_processor import ImageProcessor
from image_tagging_service.services.tag_matcher import TagMatcher


@pytest.fixture
def settings():
    return Settings(model_name="test-model", auth_enabled=False)


@pytest.fixture
def image_processor():
    return ImageProcessor(max_dimension=512)


@pytest.fixture
def tag_matcher():
    return TagMatcher(similarity_threshold=75.0)


@pytest.fixture
def mock_classifier():
    classifier = MagicMock(spec=ImageClassifier)
    classifier.is_loaded = True
    classifier.model_name = "test-model"
    classifier.classify.return_value = [
        {"path": ["Nature", "Animals", "Birds"], "confidence": 0.95},
        {"path": ["Weather", "Sunny"], "confidence": 0.8},
    ]
    return classifier


@pytest.fixture
def sample_image_bytes():
    img = Image.new("RGB", (2000, 1500), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes():
    img = Image.new("RGBA", (800, 600), color=(255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(mock_classifier, settings):
    app = create_app()

    # Override dependencies
    dependencies._classifier = mock_classifier
    dependencies._image_processor = ImageProcessor(max_dimension=512)
    dependencies._tag_matcher = TagMatcher()
    dependencies.get_settings.cache_clear()

    with TestClient(app) as c:
        yield c

    # Clean up
    dependencies._classifier = None
    dependencies._image_processor = None
    dependencies._tag_matcher = None
    dependencies.get_settings.cache_clear()
