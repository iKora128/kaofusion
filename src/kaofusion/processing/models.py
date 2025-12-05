"""ONNX model management.

Clean implementation of model loading, caching, and downloading.
No global state - models are managed via a ModelManager instance.
"""

import logging
import platform
import subprocess
import zlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


def detect_execution_providers() -> list[str]:
    """Detect the best available execution providers.

    Priority order:
    1. CUDA (NVIDIA GPU)
    2. CoreML (Apple Silicon / macOS)
    3. DirectML (Windows GPU)
    4. CPU (fallback)

    Returns:
        List of execution providers to use
    """
    available = ort.get_available_providers()
    providers: list[str] = []

    # CUDA for NVIDIA GPUs
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
        logger.info("CUDA execution provider detected")

    # CoreML for Apple Silicon / macOS
    if "CoreMLExecutionProvider" in available:
        # CoreML works best on Apple Silicon
        if platform.system() == "Darwin":
            providers.append("CoreMLExecutionProvider")
            logger.info("CoreML execution provider detected (macOS)")

    # DirectML for Windows GPU
    if "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
        logger.info("DirectML execution provider detected (Windows)")

    # Always add CPU as fallback
    providers.append("CPUExecutionProvider")

    logger.info(f"Using execution providers: {providers}")
    return providers

# Default model directory
DEFAULT_MODELS_DIR = Path.home() / ".kaofusion" / "models"

# Base URLs for model downloads
GITHUB_BASE_URL = "https://github.com/facefusion/facefusion-assets/releases/download"
HUGGINGFACE_BASE_URL = "https://huggingface.co/facefusion/facefusion-assets/resolve/main"


@dataclass
class ModelSpec:
    """Specification for an ONNX model."""

    name: str
    filename: str
    release_tag: str  # e.g., "models-3.0.0"
    hash_filename: str | None = None
    input_size: tuple[int, int] | None = None  # (width, height)
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @property
    def download_url(self) -> str:
        """Get GitHub download URL."""
        return f"{GITHUB_BASE_URL}/{self.release_tag}/{self.filename}"

    @property
    def hash_url(self) -> str | None:
        """Get hash file download URL."""
        if self.hash_filename:
            return f"{GITHUB_BASE_URL}/{self.release_tag}/{self.hash_filename}"
        return None


# Model specifications
MODELS: dict[str, ModelSpec] = {
    # Face detectors
    "retinaface_10g": ModelSpec(
        name="retinaface_10g",
        filename="retinaface_10g.onnx",
        hash_filename="retinaface_10g.hash",
        release_tag="models-3.0.0",
        input_size=(640, 640),
    ),
    # Face landmarker
    "2dfan4": ModelSpec(
        name="2dfan4",
        filename="2dfan4.onnx",
        hash_filename="2dfan4.hash",
        release_tag="models-3.0.0",
        input_size=(256, 256),
    ),
    # Face recognizer (ArcFace)
    "arcface_w600k_r50": ModelSpec(
        name="arcface_w600k_r50",
        filename="arcface_w600k_r50.onnx",
        hash_filename="arcface_w600k_r50.hash",
        release_tag="models-3.0.0",
        input_size=(112, 112),
    ),
    # Face swapper
    "inswapper_128": ModelSpec(
        name="inswapper_128",
        filename="inswapper_128.onnx",
        hash_filename="inswapper_128.hash",
        release_tag="models-3.0.0",
        input_size=(128, 128),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    ),
    "inswapper_128_fp16": ModelSpec(
        name="inswapper_128_fp16",
        filename="inswapper_128_fp16.onnx",
        hash_filename="inswapper_128_fp16.hash",
        release_tag="models-3.0.0",
        input_size=(128, 128),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    ),
    # Face occluder (for masking)
    "xseg_1": ModelSpec(
        name="xseg_1",
        filename="xseg_1.onnx",
        hash_filename="xseg_1.hash",
        release_tag="models-3.1.0",
        input_size=(256, 256),
    ),
    # Landmark helper (5->68 converter)
    "fan_68_5": ModelSpec(
        name="fan_68_5",
        filename="fan_68_5.onnx",
        hash_filename="fan_68_5.hash",
        release_tag="models-3.0.0",
        input_size=(128, 128),
    ),
    # Face enhancers
    "gfpgan_1.4": ModelSpec(
        name="gfpgan_1.4",
        filename="gfpgan_1.4.onnx",
        hash_filename="gfpgan_1.4.hash",
        release_tag="models-3.0.0",
        input_size=(512, 512),
    ),
    "codeformer": ModelSpec(
        name="codeformer",
        filename="codeformer.onnx",
        hash_filename="codeformer.hash",
        release_tag="models-3.0.0",
        input_size=(512, 512),
    ),
    "restoreformer_plus_plus": ModelSpec(
        name="restoreformer_plus_plus",
        filename="restoreformer_plus_plus.onnx",
        hash_filename="restoreformer_plus_plus.hash",
        release_tag="models-3.0.0",
        input_size=(512, 512),
    ),
    # Frame enhancer
    "real_esrgan_x4": ModelSpec(
        name="real_esrgan_x4",
        filename="real_esrgan_x4.onnx",
        hash_filename="real_esrgan_x4.hash",
        release_tag="models-3.0.0",
        input_size=None,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    ),
    # Hyperswapper family
    "hyperswap_1a_256": ModelSpec(
        name="hyperswap_1a_256",
        filename="hyperswap_1a_256.onnx",
        hash_filename="hyperswap_1a_256.hash",
        release_tag="models-3.3.0",
        input_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "hyperswap_1b_256": ModelSpec(
        name="hyperswap_1b_256",
        filename="hyperswap_1b_256.onnx",
        hash_filename="hyperswap_1b_256.hash",
        release_tag="models-3.3.0",
        input_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "hyperswap_1c_256": ModelSpec(
        name="hyperswap_1c_256",
        filename="hyperswap_1c_256.onnx",
        hash_filename="hyperswap_1c_256.hash",
        release_tag="models-3.3.0",
        input_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
}


class ModelManager:
    """Manages ONNX model loading, caching, and downloading.

    Thread-safe singleton-like manager for ONNX inference sessions.
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        execution_providers: list[str] | None = None,
    ):
        """Initialize model manager.

        Args:
            models_dir: Directory for model storage
            execution_providers: ONNX execution providers (default: CPU only)
        """
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.execution_providers = execution_providers or detect_execution_providers()
        self._sessions: dict[str, InferenceSession] = {}
        self._lock = Lock()

    def get_session(self, model_name: str) -> InferenceSession:
        """Get or create an inference session for a model.

        Downloads the model if not present.

        Args:
            model_name: Name of the model (must be in MODELS dict)

        Returns:
            ONNX InferenceSession

        Raises:
            ValueError: If model name is unknown
            RuntimeError: If model download or loading fails
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        with self._lock:
            if model_name not in self._sessions:
                model_path = self._ensure_model(model_name)
                self._sessions[model_name] = self._create_session(model_path)

            return self._sessions[model_name]

    def get_model_spec(self, model_name: str) -> ModelSpec:
        """Get model specification.

        Args:
            model_name: Name of the model

        Returns:
            ModelSpec for the model

        Raises:
            ValueError: If model name is unknown
        """
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        return MODELS[model_name]

    def _ensure_model(self, model_name: str) -> Path:
        """Ensure model is downloaded and valid.

        Args:
            model_name: Name of the model

        Returns:
            Path to the model file
        """
        spec = MODELS[model_name]
        model_path = self.models_dir / spec.filename
        hash_path = self.models_dir / spec.hash_filename if spec.hash_filename else None

        # Download hash file if needed
        if hash_path and not hash_path.exists():
            if spec.hash_url:
                self._download_file(spec.hash_url, hash_path)

        # Check if model exists and is valid
        if model_path.exists():
            if self._validate_model(model_path, hash_path):
                logger.debug(f"Model {model_name} is valid")
                return model_path
            else:
                logger.warning(f"Model {model_name} hash mismatch, re-downloading")
                model_path.unlink()

        # Download model
        logger.info(f"Downloading model: {model_name}")
        self._download_file(spec.download_url, model_path)

        # Validate after download
        if not self._validate_model(model_path, hash_path):
            raise RuntimeError(f"Downloaded model {model_name} failed validation")

        return model_path

    def _validate_model(self, model_path: Path, hash_path: Path | None) -> bool:
        """Validate model against hash file.

        Args:
            model_path: Path to model file
            hash_path: Path to hash file (optional)

        Returns:
            True if valid or no hash to check
        """
        if not model_path.exists():
            return False

        if not hash_path or not hash_path.exists():
            # No hash to validate against, assume valid if file exists
            return model_path.stat().st_size > 0

        # Read expected hash
        expected_hash = hash_path.read_text().strip()

        # Calculate actual hash using CRC32 (FaceFusion format)
        with open(model_path, "rb") as f:
            content = f.read()
        actual_hash = format(zlib.crc32(content), "08x")

        return actual_hash == expected_hash

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file using curl.

        Args:
            url: URL to download
            dest: Destination path
        """
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Use curl for download (more reliable than urllib for large files)
        cmd = [
            "curl",
            "-L",  # Follow redirects
            "-o",
            str(dest),
            "--progress-bar",
            url,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if dest.exists():
                dest.unlink()
            raise RuntimeError(f"Failed to download {url}: {e}")

    def _create_session(self, model_path: Path) -> InferenceSession:
        """Create an ONNX inference session.

        Args:
            model_path: Path to ONNX model

        Returns:
            InferenceSession
        """
        logger.info(f"Loading model: {model_path.name}")
        return InferenceSession(
            str(model_path),
            providers=self.execution_providers,
        )

    def run_inference(
        self,
        model_name: str,
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference on a model.

        Args:
            model_name: Name of the model
            inputs: Input tensors as dict

        Returns:
            List of output tensors
        """
        session = self.get_session(model_name)
        return session.run(None, inputs)

    def clear_cache(self) -> None:
        """Clear all cached sessions."""
        with self._lock:
            self._sessions.clear()

    def get_model_initializer(self, model_name: str) -> np.ndarray | None:
        """Get model initializer weights (for inswapper).

        Some models like inswapper_128 store embedding transformation
        weights in the model initializers.

        Args:
            model_name: Name of the model

        Returns:
            Initializer array or None
        """
        spec = MODELS.get(model_name)
        if not spec:
            return None

        model_path = self.models_dir / spec.filename
        if not model_path.exists():
            return None

        try:
            import onnx

            model = onnx.load(str(model_path))
            for initializer in model.graph.initializer:
                if initializer.dims == [512, 512]:
                    return np.array(onnx.numpy_helper.to_array(initializer))
        except Exception:
            pass

        return None


# Global instance (can be replaced for testing)
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def set_model_manager(manager: ModelManager) -> None:
    """Set the global model manager instance."""
    global _model_manager
    _model_manager = manager
