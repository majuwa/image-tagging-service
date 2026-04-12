import structlog
from rapidfuzz import fuzz, process

logger = structlog.get_logger()


class TagMatcher:
    def __init__(self, similarity_threshold: float = 75.0):
        self.similarity_threshold = similarity_threshold

    def flatten_tag_paths(self, tags: list[dict]) -> dict[str, list[str]]:
        """Convert hierarchical tags to flat lookup.

        Example: {'Nature > Animals > Birds': ['Nature', 'Animals', 'Birds']}
        """
        flat: dict[str, list[str]] = {}
        for tag in tags:
            path = tag["path"]
            key = " > ".join(path)
            flat[key] = path
        return flat

    def match_tags(
        self,
        suggested: list[dict],
        existing_flat: dict[str, list[str]],
        confidence_threshold: float,
    ) -> tuple[list[dict], list[dict]]:
        """Match suggested tags against existing taxonomy.

        Returns (matched, new) where matched are existing tags and new are genuinely novel.
        """
        matched: list[dict] = []
        new: list[dict] = []
        existing_keys = list(existing_flat.keys())

        for suggestion in suggested:
            suggested_path = " > ".join(suggestion["path"])
            suggested_confidence = suggestion.get("confidence", 0.5)

            if suggested_confidence < confidence_threshold:
                continue

            # Exact match
            if suggested_path in existing_flat:
                matched.append(
                    {
                        "path": existing_flat[suggested_path],
                        "confidence": suggested_confidence,
                        "is_new": False,
                    }
                )
                continue

            # Leaf-node match (last element matches exactly)
            leaf = suggestion["path"][-1]
            leaf_matches = [k for k in existing_keys if k.split(" > ")[-1].lower() == leaf.lower()]
            if leaf_matches:
                best = leaf_matches[0]
                matched.append(
                    {
                        "path": existing_flat[best],
                        "confidence": suggested_confidence * 0.95,
                        "is_new": False,
                    }
                )
                continue

            # Fuzzy match
            if existing_keys:
                result = process.extractOne(
                    suggested_path, existing_keys, scorer=fuzz.token_sort_ratio
                )
                if result and result[1] >= self.similarity_threshold:
                    matched.append(
                        {
                            "path": existing_flat[result[0]],
                            "confidence": suggested_confidence * (result[1] / 100),
                            "is_new": False,
                        }
                    )
                    continue

            # No match – genuinely new tag
            new.append(
                {
                    "path": suggestion["path"],
                    "confidence": suggested_confidence,
                    "is_new": True,
                }
            )

        return matched, new
