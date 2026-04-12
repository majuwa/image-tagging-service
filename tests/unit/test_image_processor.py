import io

from PIL import Image

from image_tagging_service.services.image_processor import ImageProcessor


class TestImageProcessorScaling:
    def test_scale_large_image(self, sample_image_bytes):
        processor = ImageProcessor(max_dimension=512)
        result = processor.scale_image(sample_image_bytes)

        assert max(result.size) <= 512
        assert result.mode == "RGB"

    def test_no_scale_small_image(self):
        img = Image.new("RGB", (200, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        processor = ImageProcessor(max_dimension=512)
        result = processor.scale_image(buf.getvalue())

        assert result.size == (200, 100)

    def test_preserves_aspect_ratio(self, sample_image_bytes):
        processor = ImageProcessor(max_dimension=500)
        result = processor.scale_image(sample_image_bytes)

        # Original is 2000x1500 (4:3 ratio)
        w, h = result.size
        original_ratio = 2000 / 1500
        result_ratio = w / h
        assert abs(original_ratio - result_ratio) < 0.01

    def test_converts_rgba_to_rgb(self, sample_png_bytes):
        processor = ImageProcessor(max_dimension=1024)
        result = processor.scale_image(sample_png_bytes)
        assert result.mode == "RGB"

    def test_custom_max_dimension_override(self, sample_image_bytes):
        processor = ImageProcessor(max_dimension=1024)
        result = processor.scale_image(sample_image_bytes, max_dimension=256)
        assert max(result.size) <= 256

    def test_exact_boundary_no_resize(self):
        img = Image.new("RGB", (512, 512), color="green")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")

        processor = ImageProcessor(max_dimension=512)
        result = processor.scale_image(buf.getvalue())
        assert result.size == (512, 512)


class TestFormatValidation:
    def test_valid_jpeg(self):
        processor = ImageProcessor()
        assert processor.validate_format("image/jpeg", ["image/jpeg", "image/png"])

    def test_valid_png(self):
        processor = ImageProcessor()
        assert processor.validate_format("image/png", ["image/jpeg", "image/png"])

    def test_invalid_format(self):
        processor = ImageProcessor()
        assert not processor.validate_format("image/gif", ["image/jpeg", "image/png"])

    def test_empty_supported_list(self):
        processor = ImageProcessor()
        assert not processor.validate_format("image/jpeg", [])
