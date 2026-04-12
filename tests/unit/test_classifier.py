from image_tagging_service.services.classifier import ImageClassifier


class TestParseTags:
    def setup_method(self):
        self.classifier = ImageClassifier(model_name="test", device="cpu")

    def test_valid_json_array(self):
        response = '[{"path": ["Nature", "Trees"], "confidence": 0.9}]'
        result = self.classifier._parse_tags(response)

        assert len(result) == 1
        assert result[0]["path"] == ["Nature", "Trees"]
        assert result[0]["confidence"] == 0.9

    def test_json_with_surrounding_text(self):
        response = 'Here are the tags:\n[{"path": ["Sky"], "confidence": 0.8}]\nDone!'
        result = self.classifier._parse_tags(response)

        assert len(result) == 1
        assert result[0]["path"] == ["Sky"]

    def test_multiple_tags(self):
        response = """[
            {"path": ["Nature", "Mountains"], "confidence": 0.95},
            {"path": ["Weather", "Cloudy"], "confidence": 0.7}
        ]"""
        result = self.classifier._parse_tags(response)

        assert len(result) == 2
        assert result[0]["path"] == ["Nature", "Mountains"]
        assert result[1]["path"] == ["Weather", "Cloudy"]

    def test_missing_confidence_defaults_to_half(self):
        response = '[{"path": ["Ocean"]}]'
        result = self.classifier._parse_tags(response)

        assert len(result) == 1
        assert result[0]["confidence"] == 0.5

    def test_no_json_returns_empty(self):
        response = "I cannot process this image"
        result = self.classifier._parse_tags(response)
        assert result == []

    def test_malformed_json_returns_empty(self):
        response = "[{invalid json}]"
        result = self.classifier._parse_tags(response)
        assert result == []

    def test_empty_string(self):
        result = self.classifier._parse_tags("")
        assert result == []

    def test_skips_entries_without_path(self):
        response = '[{"confidence": 0.9}, {"path": ["Valid"], "confidence": 0.8}]'
        result = self.classifier._parse_tags(response)

        assert len(result) == 1
        assert result[0]["path"] == ["Valid"]

    def test_path_elements_converted_to_strings(self):
        response = '[{"path": [1, 2, 3], "confidence": 0.5}]'
        result = self.classifier._parse_tags(response)

        assert result[0]["path"] == ["1", "2", "3"]

    def test_nested_json_extracts_array(self):
        response = '```json\n[{"path": ["Test"], "confidence": 0.7}]\n```'
        result = self.classifier._parse_tags(response)

        assert len(result) == 1
        assert result[0]["path"] == ["Test"]
