import base64
import io
import json

import httpx
import structlog
from PIL import Image

logger = structlog.get_logger()

# Timeout: 10s connect, 600s read (large model inference can be slow)
_DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=600.0)


class ImageClassifier:
    """Classifies images via an OpenAI-compatible vision API (LM Studio, Ollama, etc.)."""

    def __init__(
        self,
        model_name: str,
        llm_base_url: str = "http://127.0.0.1:1234",
        **_kwargs: object,
    ):
        self.model_name = model_name
        self.llm_base_url = llm_base_url.rstrip("/")
        self._loaded = False
        self._load_error: str | None = None

    def load_model(self) -> None:
        """Verify the external LLM service is reachable."""
        logger.info("checking_llm_service", url=self.llm_base_url, model=self.model_name)
        try:
            resp = httpx.get(f"{self.llm_base_url}/v1/models", timeout=10.0)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_ids = [m.get("id", "") for m in models]
            logger.info("llm_models_available", models=model_ids)
            self._loaded = True
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            logger.warning("llm_service_unavailable", error=str(exc))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def classify(
        self, image: Image.Image, existing_tags: list[dict], max_new_tags: int = 10
    ) -> list[dict]:
        """Classify an image via the external LLM vision API."""
        if not self._loaded:
            raise RuntimeError("LLM service not available. Call load_model() first.")

        prompt = self._build_prompt(existing_tags, max_new_tags)

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_data_url = f"data:image/jpeg;base64,{b64}"
        logger.info(
            "classify_image_prepared",
            image_size=f"{image.width}x{image.height}",
            jpeg_bytes=len(buf.getvalue()),
            existing_tags=len(existing_tags),
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95,
            "stream": False,
        }

        logger.info("classify_llm_request", model=self.model_name)
        resp = httpx.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        response_text = data["choices"][0]["message"]["content"]
        logger.info("llm_response", response=response_text[:1000])

        tags = self._parse_tags(response_text)
        logger.info("parsed_tags", count=len(tags), tags=tags[:5])
        return tags

    def rate(self, image: Image.Image) -> dict:
        """Rate image quality/appeal via the external LLM vision API.

        Returns a dict with ``rating`` (int 1–5) and ``reasoning`` (str).
        """
        if not self._loaded:
            raise RuntimeError("LLM service not available. Call load_model() first.")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_data_url = f"data:image/jpeg;base64,{b64}"

        prompt = (
            "Please rate this photograph on a scale from 1 to 5 stars based on:\n"
            "- Technical quality (focus sharpness, correct exposure, low noise, white balance)\n"
            "- Composition (rule of thirds, framing, balance, leading lines)\n"
            "- Overall visual impact and appeal\n\n"
            "Rating scale:\n"
            "1 = Poor (severely under/overexposed, very blurry, or badly composed)\n"
            "2 = Below average (noticeable technical or compositional issues)\n"
            "3 = Average (acceptable quality, decent composition)\n"
            "4 = Good (technically sound with strong composition)\n"
            "5 = Excellent (outstanding quality, very compelling image)\n\n"
            "Return ONLY a valid JSON object with exactly these two fields:\n"
            '- "rating": integer from 1 to 5\n'
            '- "reasoning": one short sentence (max 20 words) explaining the rating\n\n'
            "Example: {\"rating\": 4, \"reasoning\": \"Sharp focus and balanced exposure with good use of natural light.\"}\n\n"
            "JSON output:"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 128,
            "temperature": 0.3,
            "stream": False,
        }

        logger.info("rate_llm_request", model=self.model_name)
        resp = httpx.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        response_text = data["choices"][0]["message"]["content"]
        logger.info("rate_llm_response", response=response_text[:500])
        return self._parse_rating(response_text)

    def caption(self, image: Image.Image, context: dict | None = None) -> dict:
        """Generate a social media caption via the external LLM vision API.

        context keys (all optional):
            tags       : list[str] — keyword names already assigned to the photo
            location   : str       — raw GPS string, e.g. "48.8566, 2.3522"
            city       : str
            country    : str
            date_taken : str       — formatted date/time string
            camera     : str       — camera model name
        Returns a dict with ``caption`` and ``hashtags``.
        """
        if not self._loaded:
            raise RuntimeError("LLM service not available. Call load_model() first.")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_data_url = f"data:image/jpeg;base64,{b64}"

        # Build a context block from all available metadata
        ctx = context or {}

        # Language config
        lang_code = ctx.get("language", "de")
        lang_map = {
            "de": ("Deutsch",   "auf Deutsch"),
            "en": ("English",   "in English"),
            "fr": ("Français",  "en français"),
            "es": ("Español",   "en español"),
            "it": ("Italiano",  "in italiano"),
            "pt": ("Português", "em português"),
        }
        _lang_name, lang_instruction = lang_map.get(lang_code, lang_map["de"])

        context_lines: list[str] = []
        if ctx.get("tags"):
            context_lines.append("Tags: " + ", ".join(str(t) for t in ctx["tags"]))
        location_parts: list[str] = []
        for key in ("city", "country"):
            if ctx.get(key):
                location_parts.append(str(ctx[key]))
        if not location_parts and ctx.get("location"):
            location_parts.append(f"GPS {ctx['location']}")
        if location_parts:
            context_lines.append("Location: " + ", ".join(location_parts))
        if ctx.get("date_taken"):
            context_lines.append(f"Date: {ctx['date_taken']}")
        if ctx.get("camera"):
            context_lines.append(f"Camera: {ctx['camera']}")

        context_block = ""
        if context_lines:
            context_block = (
                "\n\nPhoto metadata:\n"
                + "\n".join(f"- {line}" for line in context_lines)
                + "\n"
            )

        prompt = (
            f"You are a social media expert and photographer. "
            f"Based on this image, generate an engaging Instagram caption {lang_instruction}."
            f"{context_block}\n"
            "Return ONLY a valid JSON object with exactly these two string fields:\n"
            '- "caption": an authentic, storytelling Instagram caption (2–4 sentences, '
            "conversational tone, 1–2 relevant emojis allowed, NO hashtags in this field)\n"
            '- "hashtags": 15–20 relevant hashtags as a single space-separated string, '
            "WITHOUT a leading # character (it will be added automatically). "
            "Mix popular and niche hashtags.\n\n"
            "Example structure:\n"
            '{\n'
            f'  "caption": "...",\n'
            '  "hashtags": "photography nature landscape golden_hour"\n'
            "}\n\n"
            "JSON output:"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 400,
            "temperature": 0.7,
            "stream": False,
        }

        logger.info("caption_llm_request", model=self.model_name, context_keys=list(ctx.keys()))
        resp = httpx.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        response_text = data["choices"][0]["message"]["content"]
        logger.info("caption_llm_response", response=response_text[:500])
        return self._parse_caption(response_text)

    def review(self, image: Image.Image) -> dict:
        """Generate a structured photo critique via the external LLM vision API.

        Returns a dict with keys: composition, image_quality, subject,
        editing_tips, mood, overall.
        """
        if not self._loaded:
            raise RuntimeError("LLM service not available. Call load_model() first.")

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_data_url = f"data:image/jpeg;base64,{b64}"

        prompt = (
            "You are an expert photography critic and Lightroom editing coach. "
            "Analyse the provided photograph and return a structured critique.\n\n"
            "Return ONLY a valid JSON object with exactly these six string fields "
            "(each value is 1–3 sentences in English):\n"
            '- "composition": evaluate framing, rule of thirds, leading lines, '
            "balance, use of negative space, and viewpoint.\n"
            '- "image_quality": assess technical quality — sharpness/focus, '
            "exposure (highlights, shadows, midtones), noise/grain, "
            "white balance, and colour accuracy.\n"
            '- "subject": describe the main subject(s), how well they are '
            "isolated or contextualised, and whether the eye is drawn to them.\n"
            '- "editing_tips": give specific, actionable Lightroom editing '
            "suggestions (e.g. increase exposure by +0.5 EV, boost shadows, "
            "reduce highlights, add clarity or dehaze, adjust white balance, "
            "crop tighter, apply a colour grade).\n"
            '- "mood": capture the emotional atmosphere and feeling conveyed — '
            "lighting quality, colour palette, and story the image tells.\n"
            '- "overall": one concise summary of the image\'s strongest point, '
            "main weakness, and the single most impactful improvement to make.\n\n"
            "Example structure (values replaced with real analysis):\n"
            '{\n'
            '  "composition": "...",\n'
            '  "image_quality": "...",\n'
            '  "subject": "...",\n'
            '  "editing_tips": "...",\n'
            '  "mood": "...",\n'
            '  "overall": "..."\n'
            "}\n\n"
            "JSON output:"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.5,
            "stream": False,
        }

        logger.info("review_llm_request", model=self.model_name)
        resp = httpx.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json=payload,
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        response_text = data["choices"][0]["message"]["content"]
        logger.info("review_llm_response", response=response_text[:500])
        return self._parse_review(response_text)



    @staticmethod
    def _parse_caption(response: str) -> dict:
        """Parse LLM response into a caption dict with ``caption`` and ``hashtags``."""
        text = response.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("no_caption_json_found", response=text[:200])
            return {"caption": "Ein wunderschönes Bild.", "hashtags": "#fotografie"}
        try:
            data = json.loads(text[start:end])
            caption = str(data.get("caption", "")).strip() or "Ein wunderschönes Bild."
            raw_tags = str(data.get("hashtags", "")).strip()
            # Normalise: ensure every token starts with exactly one #
            hashtags = " ".join(
                f"#{w.lstrip('#')}" for w in raw_tags.split() if w
            )
            if not hashtags:
                hashtags = "#fotografie"
            return {"caption": caption, "hashtags": hashtags}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("caption_parse_error", error=str(e), response=text[:200])
            return {"caption": "Ein wunderschönes Bild.", "hashtags": "#fotografie"}

    @staticmethod
    def _parse_review(response: str) -> dict:
        """Parse LLM response into a structured review dict."""
        text = response.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("no_review_json_found", response=text[:200])
            return {
                "composition": "Unable to analyse composition.",
                "image_quality": "Unable to assess image quality.",
                "subject": "Unable to evaluate subject.",
                "editing_tips": "Unable to provide editing tips.",
                "mood": "Unable to capture mood.",
                "overall": "Review could not be parsed.",
            }
        try:
            data = json.loads(text[start:end])
            fields = ("composition", "image_quality", "subject", "editing_tips", "mood", "overall")
            return {f: str(data.get(f, "")).strip() or f"No {f} analysis available." for f in fields}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("review_parse_error", error=str(e), response=text[:200])
            return {
                "composition": "Parse error.",
                "image_quality": "Parse error.",
                "subject": "Parse error.",
                "editing_tips": "Parse error.",
                "mood": "Parse error.",
                "overall": "Review could not be parsed from the model response.",
            }

    @staticmethod
    def _build_prompt(existing_tags: list[dict], max_new_tags: int) -> str:
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

        return (
            "Analyze this image and suggest descriptive tags for it. "
            "Tags should be hierarchical (parent > child format).\n"
            "IMPORTANT: All tag names MUST be in German (e.g. 'Natur', 'Tiere', 'Landschaft').\n"
            f"{tags_context}\n"
            "Return ONLY a valid JSON array of objects, each with:\n"
            '- "path": array of strings, one string per hierarchy level '
            '(e.g., ["Natur", "Tiere", "Vögel"]).  '
            "Do NOT join levels with '>'; each level must be a separate string.\n"
            '- "confidence": float between 0 and 1 indicating how confident you are\n\n'
            "Rules:\n"
            f"- Suggest up to {max_new_tags + len(existing_tag_strings[:20])} tags total\n"
            "- Use hierarchical categories (2-4 levels deep)\n"
            "- Be specific but not overly granular\n"
            "- Include both broad categories and specific details\n"
            "- Confidence should reflect how clearly the subject is visible\n"
            "- ALL tag names must be in German\n\n"
            "Example output:\n"
            "[\n"
            '  {"path": ["Natur", "Landschaft", "Berge"], "confidence": 0.95},\n'
            '  {"path": ["Wetter", "Sonnig"], "confidence": 0.8}\n'
            "]\n\n"
            "JSON output:"
        )

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
                    # Normalize: LLMs sometimes return "A > B > C" as a
                    # single string instead of ["A", "B", "C"].  Split any
                    # element that contains " > " into separate components.
                    normalized: list[str] = []
                    for p in tag["path"]:
                        s = str(p).strip()
                        if " > " in s:
                            normalized.extend(part.strip() for part in s.split(" > "))
                        else:
                            normalized.append(s)
                    validated.append(
                        {
                            "path": normalized,
                            "confidence": float(tag.get("confidence", 0.5)),
                        }
                    )
            return validated
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("json_parse_error", error=str(e), response=text[:200])
            return []

    @staticmethod
    def _parse_rating(response: str) -> dict:
        """Parse LLM response into a rating dict with ``rating`` and ``reasoning``."""
        text = response.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("no_rating_json_found", response=text[:200])
            return {"rating": 3, "reasoning": "Could not parse rating response."}
        try:
            data = json.loads(text[start:end])
            rating = int(data.get("rating", 3))
            rating = max(1, min(5, rating))
            reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."
            return {"rating": rating, "reasoning": reasoning}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("rating_parse_error", error=str(e), response=text[:200])
            return {"rating": 3, "reasoning": "Could not parse rating response."}
