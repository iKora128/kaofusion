"""Core type definitions for face processing.

This module defines clean, type-safe data structures for face processing.
Unlike the original facefusion which uses namedtuples and TypedDicts with
global state, we use Pydantic models and dataclasses for better validation
and IDE support.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Type aliases for numpy arrays
VisionFrame = NDArray[np.uint8]  # BGR image (H, W, 3)
Mask = NDArray[np.float32]  # Mask (H, W) values 0-1
Embedding = NDArray[np.float32]  # Face embedding vector
Landmarks5 = NDArray[np.float32]  # 5-point landmarks (5, 2)
Landmarks68 = NDArray[np.float32]  # 68-point landmarks (68, 2)
BoundingBox = NDArray[np.float32]  # [x1, y1, x2, y2]
AffineMatrix = NDArray[np.float32]  # 2x3 affine transform matrix


class Gender(str, Enum):
    """Gender classification."""

    FEMALE = "female"
    MALE = "male"


class WarpTemplate(str, Enum):
    """Face warp template types for alignment."""

    ARCFACE_112_V1 = "arcface_112_v1"
    ARCFACE_112_V2 = "arcface_112_v2"
    ARCFACE_128 = "arcface_128"
    FFHQ_512 = "ffhq_512"


class FaceDetectorModel(str, Enum):
    """Available face detector models."""

    RETINAFACE = "retinaface"
    SCRFD = "scrfd"
    YOLOFACE = "yoloface"
    YUNET = "yunet"


class FaceSwapperModel(str, Enum):
    """Available face swapper models."""

    INSWAPPER_128 = "inswapper_128"
    INSWAPPER_128_FP16 = "inswapper_128_fp16"
    HYPERSWAP_1A_256 = "hyperswap_1a_256"
    HYPERSWAP_1B_256 = "hyperswap_1b_256"
    HYPERSWAP_1C_256 = "hyperswap_1c_256"


class FaceEnhancerModel(str, Enum):
    """Face enhancement models."""

    NONE = "none"
    GFPGAN_1_4 = "gfpgan_1.4"
    CODEFORMER = "codeformer"
    RESTOREFORMER_PLUS_PLUS = "restoreformer_plus_plus"


class FrameEnhancerModel(str, Enum):
    """Frame enhancement models."""

    NONE = "none"
    REAL_ESRGAN_X4 = "real_esrgan_x4"


class MediaType(str, Enum):
    """Supported media types for jobs."""

    IMAGE = "image"
    VIDEO = "video"


class FaceMaskType(str, Enum):
    """Types of face masks."""

    BOX = "box"
    OCCLUSION = "occlusion"


class FaceSelectorMode(str, Enum):
    """Face selection modes for multiple faces."""

    ALL = "all"  # Swap all detected faces
    LARGEST = "largest"  # Swap only the largest face
    BEST_MATCH = "best_match"  # Swap face most similar to source (by embedding)
    BY_INDEX = "by_index"  # Swap specific face by detection order
    BY_INDICES = "by_indices"  # Swap multiple specific faces by indices


class OutputFormat(str, Enum):
    """Output image formats."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    preserve_audio: bool = True
    crf: int = 18
    fps: float | None = None
    encoder: str = "libx264"
    max_frames: int | None = None  # Optional guard for previews/debug
    hwaccel: str | None = None  # e.g., "cuda", "videotoolbox"
    decoder: str | None = None  # e.g., "h264_cuvid"
    redetect_interval: int = 0  # 0 = detect every frame
    max_side: int | None = None  # Downscale long edge before processing
    progress_fps_hint: float = 24.0  # Used when frame count is unknown


@dataclass
class Landmarks:
    """Facial landmarks container.

    Stores both 5-point and 68-point landmarks.
    5-point: left_eye, right_eye, nose, left_mouth, right_mouth
    68-point: detailed facial contour and features
    """

    landmarks_5: Landmarks5  # (5, 2) - essential points
    landmarks_68: Landmarks68 | None = None  # (68, 2) - detailed points

    @property
    def five(self) -> Landmarks5:
        """Get 5-point landmarks."""
        return self.landmarks_5

    @property
    def sixty_eight(self) -> Landmarks68 | None:
        """Get 68-point landmarks if available."""
        return self.landmarks_68


@dataclass
class FaceScores:
    """Detection and landmark scores."""

    detector: float  # Detection confidence (0-1)
    landmarker: float = 0.0  # Landmark confidence (0-1)


@dataclass
class Face:
    """Detected face with all associated data.

    This is the main data structure passed through the processing pipeline.
    Unlike facefusion's namedtuple, this is a proper dataclass with methods.
    """

    bbox: BoundingBox  # [x1, y1, x2, y2]
    landmarks: Landmarks  # Facial landmarks
    scores: FaceScores  # Detection scores
    embedding: Embedding | None = None  # Face recognition embedding
    embedding_norm: Embedding | None = None  # Normalized embedding
    gender: Gender | None = None  # Classified gender
    age: int | None = None  # Estimated age
    angle: int = 0  # Face rotation angle

    @property
    def width(self) -> float:
        """Bounding box width."""
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Bounding box height."""
        return float(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        """Bounding box center point."""
        return (
            float((self.bbox[0] + self.bbox[2]) / 2),
            float((self.bbox[1] + self.bbox[3]) / 2),
        )

    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height


@dataclass
class ModelInfo:
    """Information about an ONNX model."""

    name: str
    path: Path
    url: str
    hash_url: str | None = None
    size: tuple[int, int] | None = None  # Input size (width, height)
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class SwapConfig:
    """Configuration for face swap operation."""

    # Model selection
    detector_model: FaceDetectorModel = FaceDetectorModel.RETINAFACE
    swapper_model: FaceSwapperModel = FaceSwapperModel.HYPERSWAP_1A_256

    # Detection settings
    detector_score_threshold: float = 0.5
    detector_size: tuple[int, int] = (640, 640)

    # Face selection
    face_selector_mode: FaceSelectorMode = FaceSelectorMode.ALL
    face_selector_index: int | None = None  # Used when mode is BY_INDEX
    face_selector_indices: list[int] | None = None  # Used when mode is BY_INDICES

    # Mask settings
    mask_types: list[FaceMaskType] = field(
        default_factory=lambda: [FaceMaskType.BOX, FaceMaskType.OCCLUSION]
    )
    mask_blur: float = 0.3
    mask_padding: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, right, bottom, left
    preserve_mouth: bool = False  # Keep target mouth area when True

    # Output settings
    output_format: OutputFormat = OutputFormat.PNG
    output_quality: int = 95  # For JPEG/WebP

    # Swap settings
    swap_weight: float = 0.5  # Source/target blend weight

    # Enhancement settings
    face_enhancer_model: FaceEnhancerModel = FaceEnhancerModel.NONE
    face_enhancer_blend: float = 0.8  # 0-1, how much of the enhanced face to keep
    face_enhancer_weight: float = 0.5  # model-specific weight (e.g., CodeFormer fidelity)
    frame_enhancer_model: FrameEnhancerModel = FrameEnhancerModel.NONE
    frame_enhancer_blend: float = 1.0  # 0-1, blend with original frame

    # Video settings (used when target is a video)
    video: VideoConfig | None = None

    # Execution settings
    execution_providers: list[str] | None = None  # None for auto-detect


# Template landmarks for face alignment
# These are the target positions for warping faces to canonical positions
WARP_TEMPLATES: dict[WarpTemplate, NDArray[np.float32]] = {
    WarpTemplate.ARCFACE_112_V1: np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    ),
    WarpTemplate.ARCFACE_112_V2: np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    ),
    WarpTemplate.ARCFACE_128: np.array(
        [
            [46.2946, 51.6963],
            [81.5318, 51.5014],
            [64.0252, 71.7366],
            [49.5493, 92.3655],
            [78.7299, 92.2041],
        ],
        dtype=np.float32,
    ),
    WarpTemplate.FFHQ_512: np.array(
        [
            [192.98138, 239.94708],
            [318.90277, 240.1936],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118],
        ],
        dtype=np.float32,
    ),
}
