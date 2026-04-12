# Copilot Instructions for image-tagging-service

## Context
This is a FastAPI REST service for AI-powered image tagging. It uses Google Gemma 4 via HuggingFace Transformers.

## Code Style
- Python 3.11+ with full type annotations
- Pydantic v2 for all data models
- async/await for route handlers
- structlog for logging (not print or stdlib logging)
- ruff for linting and formatting (line-length=100)

## Testing
- pytest with pytest-asyncio
- Mock the LLM model in tests, never call real model
- Use httpx TestClient for API tests
- Test file naming: test_<module>.py

## Architecture
- Services are injected via FastAPI Depends()
- Settings loaded once via pydantic-settings
- Model loaded in lifespan context manager

## #AskQuestions
If anything is unclear about the codebase, architecture, or requirements, ask before making changes.
