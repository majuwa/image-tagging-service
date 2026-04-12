from image_tagging_service.services.tag_matcher import TagMatcher


class TestFlattenTagPaths:
    def test_flatten_single_tag(self):
        matcher = TagMatcher()
        tags = [{"path": ["Nature", "Animals", "Birds"]}]
        result = matcher.flatten_tag_paths(tags)
        assert result == {"Nature > Animals > Birds": ["Nature", "Animals", "Birds"]}

    def test_flatten_multiple_tags(self):
        matcher = TagMatcher()
        tags = [
            {"path": ["Nature", "Animals"]},
            {"path": ["People", "Portraits"]},
        ]
        result = matcher.flatten_tag_paths(tags)
        assert len(result) == 2
        assert "Nature > Animals" in result
        assert "People > Portraits" in result

    def test_flatten_empty(self):
        matcher = TagMatcher()
        assert matcher.flatten_tag_paths([]) == {}


class TestMatchTagsExact:
    def test_exact_match(self):
        matcher = TagMatcher()
        existing = {"Nature > Animals > Birds": ["Nature", "Animals", "Birds"]}
        suggested = [{"path": ["Nature", "Animals", "Birds"], "confidence": 0.9}]

        matched, new = matcher.match_tags(suggested, existing, 0.3)

        assert len(matched) == 1
        assert len(new) == 0
        assert matched[0]["path"] == ["Nature", "Animals", "Birds"]
        assert matched[0]["confidence"] == 0.9
        assert matched[0]["is_new"] is False


class TestMatchTagsLeaf:
    def test_leaf_match(self):
        matcher = TagMatcher()
        existing = {"Nature > Animals > Birds": ["Nature", "Animals", "Birds"]}
        suggested = [{"path": ["Wildlife", "Birds"], "confidence": 0.85}]

        matched, new = matcher.match_tags(suggested, existing, 0.3)

        assert len(matched) == 1
        assert matched[0]["path"] == ["Nature", "Animals", "Birds"]
        assert matched[0]["confidence"] == 0.85 * 0.95

    def test_leaf_match_case_insensitive(self):
        matcher = TagMatcher()
        existing = {"Nature > Animals > Birds": ["Nature", "Animals", "Birds"]}
        suggested = [{"path": ["Wildlife", "birds"], "confidence": 0.8}]

        matched, new = matcher.match_tags(suggested, existing, 0.3)

        assert len(matched) == 1
        assert matched[0]["path"] == ["Nature", "Animals", "Birds"]


class TestMatchTagsFuzzy:
    def test_fuzzy_match_high_similarity(self):
        matcher = TagMatcher(similarity_threshold=70.0)
        existing = {"Nature > Animals > Birds": ["Nature", "Animals", "Birds"]}
        suggested = [{"path": ["Nature", "Animal", "Bird"], "confidence": 0.8}]

        matched, new = matcher.match_tags(suggested, existing, 0.3)

        assert len(matched) == 1
        assert matched[0]["is_new"] is False


class TestMatchTagsNew:
    def test_no_match_returns_new(self):
        matcher = TagMatcher(similarity_threshold=90.0)
        existing = {"Nature > Animals > Birds": ["Nature", "Animals", "Birds"]}
        suggested = [{"path": ["Architecture", "Modern", "Skyscraper"], "confidence": 0.9}]

        matched, new = matcher.match_tags(suggested, existing, 0.3)

        assert len(matched) == 0
        assert len(new) == 1
        assert new[0]["path"] == ["Architecture", "Modern", "Skyscraper"]
        assert new[0]["is_new"] is True

    def test_empty_existing_all_new(self):
        matcher = TagMatcher()
        suggested = [
            {"path": ["Nature", "Trees"], "confidence": 0.7},
            {"path": ["Weather", "Rain"], "confidence": 0.6},
        ]

        matched, new = matcher.match_tags(suggested, {}, 0.3)

        assert len(matched) == 0
        assert len(new) == 2


class TestConfidenceFiltering:
    def test_below_threshold_filtered(self):
        matcher = TagMatcher()
        existing = {"Nature > Animals": ["Nature", "Animals"]}
        suggested = [{"path": ["Nature", "Animals"], "confidence": 0.1}]

        matched, new = matcher.match_tags(suggested, existing, 0.5)

        assert len(matched) == 0
        assert len(new) == 0

    def test_at_threshold_included(self):
        matcher = TagMatcher()
        existing = {"Nature > Animals": ["Nature", "Animals"]}
        suggested = [{"path": ["Nature", "Animals"], "confidence": 0.5}]

        matched, new = matcher.match_tags(suggested, existing, 0.5)

        assert len(matched) == 1
