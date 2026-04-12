# Image Tagging Service

AI-powered image classification and hierarchical tag suggestion service using Google Gemma 4 multimodal model via HuggingFace Transformers.

## Features

- **Image Upload & Analysis**: Upload JPEG, PNG, or WebP images for AI-powered tag suggestions
- **Hierarchical Tags**: Supports multi-level tag hierarchies (e.g., `Nature > Animals > Birds`)
- **Existing Tag Matching**: Matches AI suggestions against your existing tag taxonomy using exact, leaf-node, and fuzzy matching
- **New Tag Discovery**: Suggests genuinely new tags for subjects not covered by existing taxonomy
- **Server-Side Image Scaling**: Automatically scales images before model processing for optimal performance
- **Optional API Key Auth**: Protect endpoints with simple API key authentication
- **OpenAPI 3.1 Documentation**: Full interactive API docs at `/docs` and `/redoc`
- **Structured Logging**: JSON-formatted structured logs via structlog

## Quick Start

### Prerequisites

- Python 3.11+
- A HuggingFace-compatible GPU (recommended) or CPU

### Installation

```bash
# Clone and install
cd image-tagging-service
pip install -e ".[dev]"

# Copy and edit environment config
cp .env.example .env
```

### Running

```bash
# Development server with auto-reload
uvicorn image_tagging_service.main:app --reload

# Or via the entry point
image-tagging-service
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "image=@photo.jpg" \
  -F 'existing_tags=[{"path": ["Nature", "Animals", "Birds"]}, {"path": ["Nature", "Landscapes"]}]' \
  -F "max_new_tags=10" \
  -F "confidence_threshold=0.3"
```

## Configuration

All settings are configured via environment variables prefixed with `ITS_`:

| Variable | Default | Description |
|---|---|---|
| `ITS_HOST` | `0.0.0.0` | Server bind address |
| `ITS_PORT` | `8000` | Server port |
| `ITS_DEBUG` | `false` | Enable debug mode (auto-reload) |
| `ITS_MODEL_NAME` | `google/gemma-4-4b-it` | HuggingFace model identifier |
| `ITS_DEVICE` | `auto` | Device for inference (`auto`, `cpu`, `cuda`, `mps`) |
| `ITS_MAX_IMAGE_DIMENSION` | `1024` | Max dimension for image scaling |
| `ITS_MAX_UPLOAD_SIZE_MB` | `50` | Maximum upload file size in MB |
| `ITS_AUTH_ENABLED` | `false` | Enable API key authentication |
| `ITS_API_KEY` | *(none)* | API key (required if auth enabled) |
| `ITS_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `ITS_LOG_FORMAT` | `json` | Log format (`json` or `console`) |

## API Reference

### `POST /api/v1/classify`

Classify an image and get hierarchical tag suggestions.

**Request**: `multipart/form-data`
- `image` (file, required): Image file (JPEG, PNG, WebP)
- `existing_tags` (string, optional): JSON array of `{"path": [...]}` objects
- `max_new_tags` (int, optional): Max new tags to suggest (1–50, default: 10)
- `confidence_threshold` (float, optional): Minimum confidence (0.0–1.0, default: 0.3)

**Response**: `ClassifyResponse`
```json
{
  "matched_tags": [{"path": ["Nature", "Animals", "Birds"], "confidence": 0.95, "is_new": false}],
  "new_tags": [{"path": ["Architecture", "Modern"], "confidence": 0.72, "is_new": true}],
  "processing_time_ms": 1234,
  "model_name": "google/gemma-4-4b-it"
}
```

### `GET /api/v1/health`

Health check endpoint.

### `GET /api/v1/models/info`

Returns model and configuration information.

## Development

### Setup

```bash
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=image_tagging_service

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Linting & Formatting

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix lint errors
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

## License

MIT
