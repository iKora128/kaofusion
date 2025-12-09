"""ONNX model management.

Clean implementation of model loading, caching, and downloading.
No global state - models are managed via a ModelManager instance.
"""

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download

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

# Hugging Face repo for models
HF_REPO_ID = "longisland3/kaofusion-models"


@dataclass
class ModelSpec:
    """Specification for an ONNX model."""

    name: str
    filename: str
    input_size: tuple[int, int] | None = None  # (width, height)
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)


# Model specifications
MODELS: dict[str, ModelSpec] = {
    # Face detector
    "retinaface_10g": ModelSpec(
        name="retinaface_10g",
        filename="retinaface_10g.onnx",
        input_size=(640, 640),
    ),
    # Face landmarker
    "2dfan4": ModelSpec(
        name="2dfan4",
        filename="2dfan4.onnx",
        input_size=(256, 256),
    ),
    # Face recognizer (ArcFace)
    "arcface_w600k_r50": ModelSpec(
        name="arcface_w600k_r50",
        filename="arcface_w600k_r50.onnx",
        input_size=(112, 112),
    ),
    # Face swapper
    "inswapper_128": ModelSpec(
        name="inswapper_128",
        filename="inswapper_128.onnx",
        input_size=(128, 128),
    ),
    "inswapper_128_fp16": ModelSpec(
        name="inswapper_128_fp16",
        filename="inswapper_128_fp16.onnx",
        input_size=(128, 128),
    ),
    # Face occluder (for masking)
    "xseg_1": ModelSpec(
        name="xseg_1",
        filename="xseg_1.onnx",
        input_size=(256, 256),
    ),
    # Landmark helper (5->68 converter)
    "fan_68_5": ModelSpec(
        name="fan_68_5",
        filename="fan_68_5.onnx",
        input_size=(128, 128),
    ),
    # Face enhancers
    "gfpgan_1.4": ModelSpec(
        name="gfpgan_1.4",
        filename="gfpgan_1.4.onnx",
        input_size=(512, 512),
    ),
    "codeformer": ModelSpec(
        name="codeformer",
        filename="codeformer.onnx",
        input_size=(512, 512),
    ),
    "restoreformer_plus_plus": ModelSpec(
        name="restoreformer_plus_plus",
        filename="restoreformer_plus_plus.onnx",
        input_size=(512, 512),
    ),
    # Frame enhancer
    "real_esrgan_x4": ModelSpec(
        name="real_esrgan_x4",
        filename="real_esrgan_x4.onnx",
        input_size=None,
    ),
    # Hyperswapper family
    "hyperswap_1a_256": ModelSpec(
        name="hyperswap_1a_256",
        filename="hyperswap_1a_256.onnx",
        input_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "hyperswap_1b_256": ModelSpec(
        name="hyperswap_1b_256",
        filename="hyperswap_1b_256.onnx",
        input_size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
    "hyperswap_1c_256": ModelSpec(
        name="hyperswap_1c_256",
        filename="hyperswap_1c_256.onnx",
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
        """Ensure model is downloaded.

        Uses huggingface_hub for efficient downloading with caching.

        Args:
            model_name: Name of the model

        Returns:
            Path to the model file
        """
        spec = MODELS[model_name]
        model_path = self.models_dir / spec.filename

        # Check if model already exists locally
        if model_path.exists() and model_path.stat().st_size > 0:
            logger.debug(f"Model {model_name} found locally")
            return model_path

        # Download from Hugging Face
        logger.info(f"Downloading model: {model_name}")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=spec.filename,
            local_dir=self.models_dir,
            local_dir_use_symlinks=False,
        )

        return Path(downloaded_path)

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
