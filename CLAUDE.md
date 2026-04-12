# CLAUDE.md - Image Tagging Service

## Project Overview
FastAPI REST service for AI-powered image tagging using Gemma 4 multimodal model.

## Tech Stack
- Python 3.11+, FastAPI, Pydantic v2
- HuggingFace Transformers (Gemma 4)
- Pillow for image processing, rapidfuzz for tag matching
- pytest for testing, ruff for linting

## Key Commands
- `pip install -e ".[dev]"` — Install with dev deps
- `pytest` — Run tests
- `ruff check src/ tests/` — Lint
- `ruff format src/ tests/` — Format
- `uvicorn image_tagging_service.main:app --reload` — Dev server

## Architecture
- `src/image_tagging_service/main.py` — App factory + lifespan
- `src/image_tagging_service/config.py` — Pydantic Settings (env: ITS_*)
- `src/image_tagging_service/api/routes/` — FastAPI route handlers
- `src/image_tagging_service/services/` — Business logic (classifier, image_processor, tag_matcher)
- `tests/` — pytest tests (unit/ and integration/)

## Conventions
- All settings via environment variables prefixed ITS_
- Structured logging via structlog
- Type hints everywhere, Pydantic models for API
- ruff for linting+formatting
