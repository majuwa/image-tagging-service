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

    def test_classify_returns_503_when_model_not_loaded(
        self, client, mock_classifier, sample_image_bytes
    ):
        mock_classifier.is_loaded = False
        mock_classifier._load_error = "401 Unauthorized — gated model requires HF token"
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"]

    def test_classify_oversized_image_returns_413(self, client):
        """Verify that images exceeding max_upload_size_mb are rejected."""
        # Create a minimal JPEG header followed by padding to exceed the limit.
        # The default max_upload_size_mb is 50, but in tests we override settings
        # to a tiny value to avoid allocating 50 MB of memory.
        from image_tagging_service.api import dependencies

        dependencies.get_settings.cache_clear()

        import os

        os.environ["ITS_MAX_UPLOAD_SIZE_MB"] = "1"
        try:
            dependencies.get_settings.cache_clear()

            # Build a valid JPEG that is >1 MB
            img = Image.new("RGB", (2000, 2000), color="white")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=100)
            # Pad to exceed 1 MB
            data = buf.getvalue()
            if len(data) < 1 * 1024 * 1024 + 1:
                data = data + b"\x00" * (1 * 1024 * 1024 + 1 - len(data))
            response = client.post(
                "/api/v1/classify",
                files={"image": ("big.jpg", data, "image/jpeg")},
            )
            assert response.status_code == 413
            assert "exceeds" in response.json()["detail"]
        finally:
            os.environ.pop("ITS_MAX_UPLOAD_SIZE_MB", None)
            dependencies.get_settings.cache_clear()

    def test_classify_with_custom_parameters(self, client, sample_image_bytes):
        """Verify custom max_new_tags and confidence_threshold are accepted."""
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={
                "existing_tags": "[]",
                "max_new_tags": "5",
                "confidence_threshold": "0.8",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # With threshold 0.8, only tags >= 0.8 confidence should appear
        for tag in data["new_tags"]:
            assert tag["confidence"] >= 0.8

    def test_classify_png_image(self, client, sample_png_bytes):
        """Verify PNG images are accepted and processed correctly."""
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.png", sample_png_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "matched_tags" in data
        assert "new_tags" in data

    def test_classify_response_structure(self, client, sample_image_bytes):
        """Verify the full response schema matches ClassifyResponse."""
        response = client.post(
            "/api/v1/classify",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["matched_tags"], list)
        assert isinstance(data["new_tags"], list)
        assert isinstance(data["processing_time_ms"], int)
        assert isinstance(data["model_name"], str)

        # Verify tag structure
        for tag in data["new_tags"]:
            assert isinstance(tag["path"], list)
            assert isinstance(tag["confidence"], float)
            assert isinstance(tag["is_new"], bool)
            assert all(isinstance(p, str) for p in tag["path"])


class TestAuthMiddleware:
    def test_auth_enabled_rejects_without_key(self):
        """Test that auth middleware rejects requests without API key."""
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        from image_tagging_service.main import create_app
        from image_tagging_service.services.classifier import ImageClassifier

        prev_classifier = dependencies._classifier
        prev_processor = dependencies._image_processor
        prev_matcher = dependencies._tag_matcher

        classifier_mock = MagicMock(spec=ImageClassifier)
        classifier_mock.is_loaded = True
        classifier_mock.model_name = "test-model"
        classifier_mock._load_error = None

        dependencies._classifier = classifier_mock
        dependencies._image_processor = ImageProcessor(max_dimension=512)
        dependencies._tag_matcher = TagMatcher()
        dependencies.get_settings.cache_clear()

        import os

        os.environ["ITS_AUTH_ENABLED"] = "true"
        os.environ["ITS_API_KEY"] = "test-secret-key"
        try:
            auth_app = create_app()

            with TestClient(auth_app) as c:
                # Health should still work without auth
                response = c.get("/api/v1/health")
                assert response.status_code == 200

                # Classify without key should be rejected
                img = Image.new("RGB", (100, 100), color="red")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                response = c.post(
                    "/api/v1/classify",
                    files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
                )
                assert response.status_code == 401
                assert "Invalid or missing API key" in response.json()["detail"]

                # Classify with wrong key should be rejected
                response = c.post(
                    "/api/v1/classify",
                    files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
                    headers={"X-API-Key": "wrong-key"},
                )
                assert response.status_code == 401

                # Classify with correct key should succeed
                response = c.post(
                    "/api/v1/classify",
                    files={"image": ("test.jpg", buf.getvalue(), "image/jpeg")},
                    headers={"X-API-Key": "test-secret-key"},
                )
                assert response.status_code == 200

                # Docs endpoint should be accessible without auth
                response = c.get("/docs")
                assert response.status_code == 200
        finally:
            os.environ.pop("ITS_AUTH_ENABLED", None)
            os.environ.pop("ITS_API_KEY", None)
            dependencies._classifier = prev_classifier
            dependencies._image_processor = prev_processor
            dependencies._tag_matcher = prev_matcher
            dependencies.get_settings.cache_clear()
