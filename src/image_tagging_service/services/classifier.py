import json

import structlog
import torch
from PIL import Image

logger = structlog.get_logger()


class ImageClassifier:
    def __init__(self, model_name: str, device: str = "auto", hf_token: str | None = None):
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.model = None
        self.processor = None
        self._loaded = False
        self._load_error: str | None = None

    def load_model(self) -> None:
        """Load the model and processor. Called on startup."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("loading_model", model=self.model_name)

        device_map = self.device if self.device != "auto" else "auto"
        token = self.hf_token or None

        self.processor = AutoProcessor.from_pretrained(self.model_name, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        self._loaded = True
        self._load_error = None
        logger.info("model_loaded", model=self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def classify(
        self, image: Image.Image, existing_tags: list[dict], max_new_tags: int = 10
    ) -> list[dict]:
        """Classify an image and return tag suggestions."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        existing_tag_strings = [" > ".join(tag["path"]) for tag in existing_tags]

        tags_context = ""
        if existing_tag_strings:
            tag_lines = "\n".join(f"- {t}" for t in existing_tag_strings[:200])
            tags_context = (
                "\nHere are the existing tags in the user's library "
                "(hierarchical, separated by ' > '):\n"
                f"{tag_lines}\n\n"
                "Prefer matching existing tags when appropriate. "
                "You may also suggest new tags if the image contains subjects "
                "not covered by existing tags.\n"
            )

        prompt = (
            "Analyze this image and suggest descriptive tags for it. "
            "Tags should be hierarchical (parent > child format).\n"
            f"{tags_context}\n"
            "Return ONLY a valid JSON array of objects, each with:\n"
            '- "path": array of strings representing the hierarchical tag path '
            '(e.g., ["Nature", "Animals", "Birds"])\n'
            '- "confidence": float between 0 and 1 indicating how confident you are\n\n'
            "Rules:\n"
            f"- Suggest up to {max_new_tags + len(existing_tag_strings[:20])} tags total\n"
            "- Use hierarchical categories (2-4 levels deep)\n"
            "- Be specific but not overly granular\n"
            "- Include both broad categories and specific details\n"
            "- Confidence should reflect how clearly the subject is visible\n\n"
            "Example output:\n"
            "[\n"
            '  {"path": ["Nature", "Landscapes", "Mountains"], "confidence": 0.95},\n'
            '  {"path": ["Weather", "Sunny"], "confidence": 0.8}\n'
            "]\n\n"
            "JSON output:"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = self.processor.decode(generated, skip_special_tokens=True).strip()

        logger.debug("llm_response", response=response_text)

        return self._parse_tags(response_text)

    def _parse_tags(self, response: str) -> list[dict]:
        """Parse LLM response into structured tags."""
        text = response.strip()

        start = text.find("[")
        end = text.rfind("]") + 1

        if start == -1 or end == 0:
            logger.warning("no_json_found", response=text[:200])
            return []

        try:
            tags = json.loads(text[start:end])
            validated = []
            for tag in tags:
                if isinstance(tag, dict) and "path" in tag and isinstance(tag["path"], list):
                    validated.append(
                        {
                            "path": [str(p) for p in tag["path"]],
                            "confidence": float(tag.get("confidence", 0.5)),
                        }
                    )
            return validated
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("json_parse_error", error=str(e), response=text[:200])
            return []
