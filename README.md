# KaoFusion

Face Swap API with FastAPI + daisyUI

## Features

- Clean, modern architecture (no global state)
- Async-first FastAPI backend
- Vanilla JS + daisyUI frontend
- HyperSwap / InSwapper face swapper selection
- GFPGAN / CodeFormer / RestoreFormer++ face enhancement
- Real-ESRGAN frame upscaling
- Image & video support with ffmpeg audio passthrough
- Automatic GPU acceleration detection

## Requirements

- Python 3.10+
- uv (package manager)
- ffmpeg (for video processing)

## Installation

```bash
# Basic installation (CPU + CoreML on Mac)
uv sync

# With NVIDIA CUDA support
uv sync --extra cuda

# With Windows DirectML support
uv sync --extra directml
```

## GPU Acceleration

KaoFusion automatically detects and uses the best available execution provider:

| Platform | Provider | Package |
|----------|----------|---------|
| macOS (Apple Silicon) | CoreML | `onnxruntime` (default) |
| NVIDIA GPU | CUDA | `uv sync --extra cuda` |
| Windows GPU | DirectML | `uv sync --extra directml` |
| CPU (fallback) | CPU | `onnxruntime` (default) |

## Usage

```bash
# Start server (default port 8000)
uv run python -m kaofusion.main

# Custom port
KAOFUSION_PORT=8080 uv run python -m kaofusion.main
```

Open http://localhost:8000/ in your browser.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Health check |
| `/api/upload` | POST | Upload image/video |
| `/api/swap` | POST | Execute face swap |
| `/api/jobs/{id}` | GET | Job status |
| `/api/jobs/{id}/stream` | GET | SSE progress stream |

## Models

Models are downloaded automatically on first use to `~/.kaofusion/models/`:

| Model | Purpose | Size |
|-------|---------|------|
| retinaface_10g | Face detection | ~30MB |
| 2dfan4 | Landmark detection | ~5MB |
| arcface_w600k_r50 | Face recognition | ~250MB |
| hyperswap_1a_256 / inswapper_128 | Face swap | ~500-600MB |
| gfpgan_1.4 / codeformer | Face enhancement | ~200-400MB |
| real_esrgan_x4 | Frame enhancement | ~120MB |

## Project Structure

```
kaofusion/
├── src/kaofusion/
│   ├── main.py           # FastAPI entry point
│   ├── config.py         # Settings
│   ├── api/              # API routes & schemas
│   ├── services/         # Business logic
│   ├── processing/       # Face processing
│   └── templates/        # Jinja2 + daisyUI
├── pyproject.toml
└── uv.lock
```

## License

MIT
