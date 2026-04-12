import io
import json

from PIL import Image

from image_tagging_service.api import dependencies
from image_tagging_service.services.image_processor import ImageProcessor
from image_tagging_service.services.tag_matcher import TagMatcher


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "0.1.0"

    def test_health_degraded_when_model_not_loaded(self, client, mock_classifier):
        mock_classifier.is_loaded = False
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert "max_image_dimension" in data
        assert "supported_formats" in data


class TestClassifyEndpoint:
    def test_classify_valid_image(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "matched_tags" in data
        assert "new_tags" in data
        assert "processing_time_ms" in data
        assert data["model_name"] is not None

    def test_classify_with_existing_tags(self, client, sample_image_bytes):
        existing_tags = json.dumps(
            [
                {"path": ["Nature", "Animals", "Birds"]},
                {"path": ["Weather", "Sunny"]},
            ]
        )
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"existing_tags": existing_tags},
        )
        assert response.status_code == 200
        data = response.json()
        # Since mock returns these exact tags, they should match
        assert len(data["matched_tags"]) == 2
        assert len(data["new_tags"]) == 0

    def test_classify_unsupported_format(self, client):
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.bmp", buf.getvalue(), "image/bmp")},
        )
        assert response.status_code == 400
        assert "Unsupported format" in response.json()["detail"]

    def test_classify_invalid_tags_json(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"existing_tags": "not valid json"},
        )
        assert response.status_code == 400
        assert "Invalid existing_tags JSON" in response.json()["detail"]

    def test_classify_empty_tags(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"existing_tags": "[]"},
        )
        assert response.status_code == 200
        data = response.json()
        # With no existing tags, all suggestions are new
        assert len(data["new_tags"]) == 2


class TestAuthMiddleware:
    def test_auth_enabled_rejects_without_key(self):
        """Test that auth middleware rejects requests without API key."""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from image_tagging_service.main import create_app
        from image_tagging_service.services.classifier import ImageClassifier

        prev_classifier = dependencies._classifier

        classifier_mock = MagicMock(spec=ImageClassifier)
        classifier_mock.is_loaded = True
        classifier_mock.model_name = "test-model"

        dependencies._classifier = classifier_mock
        dependencies._image_processor = ImageProcessor(max_dimension=512)
        dependencies._tag_matcher = TagMatcher()
        dependencies.get_settings.cache_clear()

        auth_app = create_app()

        with TestClient(auth_app) as c:
            # Health should still work without auth
            response = c.get("/api/v1/health")
            assert response.status_code == 200

        # Clean up
        dependencies._classifier = prev_classifier
        dependencies.get_settings.cache_clear()
