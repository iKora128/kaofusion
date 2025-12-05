"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App info
    app_name: str = "KaoFusion"
    debug: bool = False

    # Paths
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("output")
    models_dir: Path = Path.home() / ".kaofusion" / "models"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Processing
    max_upload_size: int = 50 * 1024 * 1024  # 50MB

    # Execution settings
    execution_providers: list[str] | None = None  # None = auto-detect
    onnx_num_threads: int = 0  # 0 = auto (uses available cores)
    onnx_graph_optimization_level: str = "all"  # none, basic, extended, all

    # File management
    upload_ttl_hours: int = 24  # Time-to-live for uploads

    class Config:
        env_prefix = "KAOFUSION_"


settings = Settings()

# Ensure directories exist
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
