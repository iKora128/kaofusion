"""KaoFusion application entry point."""

import logging
import threading
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from kaofusion.api.routes import benchmark, health, jobs, swap, uploads
from kaofusion.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preload_models():
    """Preload essential models in background thread."""
    from kaofusion.processing.models import get_model_manager

    logger.info("Preloading models...")
    manager = get_model_manager()

    # Essential models for face swap
    essential_models = [
        "retinaface_10g",  # Face detector
        "2dfan4",          # Landmark detector
        "arcface_w600k_r50",  # Face recognizer
        "xseg_1",          # Occlusion mask
        "hyperswap_1a_256",  # Default swapper model
    ]

    for model_name in essential_models:
        try:
            logger.info(f"Preloading model: {model_name}")
            manager.get_session(model_name)
        except Exception as e:
            logger.warning(f"Failed to preload {model_name}: {e}")

    logger.info("Model preloading complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - preload models on startup."""
    # Start model preloading in background thread
    preload_thread = threading.Thread(target=preload_models, daemon=True)
    preload_thread.start()
    yield
    # Cleanup (if needed)


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Face swap web application",
    version="0.1.0",
    lifespan=lifespan,
)

# Get base directory
BASE_DIR = Path(__file__).parent

# Mount static files
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Include routers
app.include_router(health.router)
app.include_router(uploads.router)
app.include_router(swap.router)
app.include_router(jobs.router)
app.include_router(benchmark.router)


@app.get("/")
async def index():
    """Render main application page."""
    from fastapi.responses import HTMLResponse

    template = templates.get_template("index.html")
    return HTMLResponse(template.render({"app_name": settings.app_name}))


def run():
    """Run the application server."""
    logger.info(f"Starting {settings.app_name} on {settings.host}:{settings.port}")
    uvicorn.run(
        "kaofusion.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
