# Architecture

## System Overview

The Image Tagging Service is a REST API that accepts images along with a user's existing hierarchical tag taxonomy, analyzes images using a multimodal LLM (Google Gemma 4), and returns both matched existing tags and suggested new tags.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Auth         │  │  Health      │  │  Classify        │  │
│  │  Middleware   │  │  Routes      │  │  Routes          │  │
│  └──────┬───────┘  └──────────────┘  └────────┬─────────┘  │
│         │                                      │            │
│         ▼                                      ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Dependencies Layer                   │   │
│  │        (Dependency Injection via FastAPI Depends)     │   │
│  └──────────────────────────────────────────────────────┘   │
│         │                  │                   │            │
│         ▼                  ▼                   ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Image        │  │  Image       │  │  Tag             │  │
│  │  Classifier   │  │  Processor   │  │  Matcher         │  │
│  │  (Gemma 4)    │  │  (Pillow)    │  │  (rapidfuzz)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Request Received**: Client sends a `multipart/form-data` POST with an image file and optional JSON payload of existing tags.

2. **Authentication** (optional): If enabled, the `ApiKeyMiddleware` validates the `X-API-Key` header. Health and docs endpoints are exempt.

3. **Validation**: The route handler validates the content type, file size, and existing tags JSON format.

4. **Image Processing**: `ImageProcessor` loads the image, converts RGBA/P to RGB, and scales it down to fit within `max_image_dimension` while preserving aspect ratio.

5. **Classification**: `ImageClassifier` constructs a prompt that includes the existing tag taxonomy as context, sends the image + prompt to the Gemma 4 multimodal model, and parses the JSON response.

6. **Tag Matching**: `TagMatcher` compares the LLM's suggestions against the existing taxonomy using three strategies:
   - **Exact match**: Full path string comparison
   - **Leaf match**: Compare only the last element (case-insensitive)
   - **Fuzzy match**: Token sort ratio via rapidfuzz (configurable threshold)

7. **Response**: Returns matched existing tags and genuinely new tag suggestions, each with confidence scores.

## Technology Choices

| Component | Technology | Rationale |
|---|---|---|
| Web Framework | FastAPI | Async-native, automatic OpenAPI generation, Pydantic integration |
| LLM | Google Gemma 4 (via HF Transformers) | Open-weight multimodal model, runs locally, no API costs |
| Image Processing | Pillow | Industry standard, efficient, lightweight |
| Tag Matching | rapidfuzz | C++ backed fuzzy matching, fast on large tag sets |
| Configuration | pydantic-settings | Type-safe env config, validation, `.env` support |
| Logging | structlog | Structured JSON logging, context variables |
| Testing | pytest + httpx | Industry standard, async support, TestClient |

## Scaling Considerations

- **Model Loading**: The model is loaded once during application startup (lifespan) and shared across requests. This avoids per-request loading overhead.

- **Image Scaling**: Images are scaled server-side before being sent to the model, reducing GPU memory usage and inference time.

- **Singleton Services**: `ImageClassifier`, `ImageProcessor`, and `TagMatcher` are singletons managed via the dependency injection layer.

- **Horizontal Scaling**: Multiple instances can be run behind a load balancer. Each instance loads its own model copy. For GPU-constrained environments, consider a model serving layer (e.g., vLLM, TGI).

- **Large Tag Sets**: The tag matching uses efficient string operations and rapidfuzz's C++ backend. Tag sets up to 5000 entries are supported per request.

## Security Model

- **Optional API Key Auth**: When `ITS_AUTH_ENABLED=true`, all endpoints except `/api/v1/health`, `/docs`, `/openapi.json`, and `/redoc` require a valid `X-API-Key` header.

- **Input Validation**: All inputs are validated via Pydantic models. File size limits, supported format checks, and tag count limits prevent abuse.

- **No Persistent Storage**: The service is stateless — it does not store images or results. All processing is ephemeral.
