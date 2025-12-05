"""Job schemas."""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FaceDetectorModelSchema(str, Enum):
    """Available face detector models."""

    RETINAFACE = "retinaface"
    SCRFD = "scrfd"
    YOLOFACE = "yoloface"
    YUNET = "yunet"


class FaceSwapperModelSchema(str, Enum):
    """Available face swapper models."""

    INSWAPPER_128 = "inswapper_128"
    INSWAPPER_128_FP16 = "inswapper_128_fp16"
    HYPERSWAP_1A_256 = "hyperswap_1a_256"
    HYPERSWAP_1B_256 = "hyperswap_1b_256"
    HYPERSWAP_1C_256 = "hyperswap_1c_256"


class FaceMaskTypeSchema(str, Enum):
    """Types of face masks."""

    BOX = "box"
    OCCLUSION = "occlusion"


class FaceSelectorModeSchema(str, Enum):
    """Face selection modes for multiple faces."""

    ALL = "all"  # Swap all detected faces
    LARGEST = "largest"  # Swap only the largest face
    BEST_MATCH = "best_match"  # Swap face most similar to source (by embedding)
    BY_INDEX = "by_index"  # Swap specific face by detection order
    BY_INDICES = "by_indices"  # Swap multiple specific faces by indices


class OutputFormatSchema(str, Enum):
    """Output image formats."""

    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"


class FaceEnhancerModelSchema(str, Enum):
    """Face enhancement models."""

    NONE = "none"
    GFPGAN_1_4 = "gfpgan_1.4"
    CODEFORMER = "codeformer"
    RESTOREFORMER_PLUS_PLUS = "restoreformer_plus_plus"


class FrameEnhancerModelSchema(str, Enum):
    """Frame enhancement models."""

    NONE = "none"
    REAL_ESRGAN_X4 = "real_esrgan_x4"


class MediaTypeSchema(str, Enum):
    """Job media type."""

    IMAGE = "image"
    VIDEO = "video"


class VideoConfigSchema(BaseModel):
    """Video processing options."""

    preserve_audio: bool = Field(
        default=True,
        description="Keep original audio track if available",
    )
    crf: int = Field(
        default=18,
        ge=0,
        le=51,
        description="CRF for ffmpeg encoder (lower = higher quality)",
    )
    fps: float | None = Field(
        default=None,
        ge=1.0,
        description="Optional output FPS override. If null, keep source FPS.",
    )
    encoder: str = Field(
        default="libx264",
        description="FFmpeg video encoder to use",
    )
    max_frames: int | None = Field(
        default=None,
        ge=1,
        description="Optional frame cap for previews/debug",
    )


class SwapConfigSchema(BaseModel):
    """Configuration for face swap operation (API schema)."""

    # Model selection
    detector_model: FaceDetectorModelSchema = Field(
        default=FaceDetectorModelSchema.RETINAFACE,
        description="Face detector model to use",
    )
    swapper_model: FaceSwapperModelSchema = Field(
        default=FaceSwapperModelSchema.HYPERSWAP_1A_256,
        description="Face swapper model to use",
    )

    # Detection settings
    detector_score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence threshold",
    )
    detector_size: tuple[int, int] = Field(
        default=(640, 640),
        description="Input size for face detector (width, height)",
    )

    # Face selection
    face_selector_mode: FaceSelectorModeSchema = Field(
        default=FaceSelectorModeSchema.ALL,
        description="How to select which faces to swap in target image",
    )
    face_selector_index: int | None = Field(
        default=None,
        ge=0,
        description="Face index to swap when mode is BY_INDEX (0-based)",
    )
    face_selector_indices: list[int] | None = Field(
        default=None,
        description="Face indices to swap when mode is BY_INDICES (0-based list)",
    )

    # Mask settings
    mask_types: list[FaceMaskTypeSchema] = Field(
        default=[FaceMaskTypeSchema.BOX, FaceMaskTypeSchema.OCCLUSION],
        description="Types of masks to apply",
    )
    mask_blur: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Mask edge blur amount",
    )
    mask_padding: tuple[int, int, int, int] = Field(
        default=(0, 0, 0, 0),
        description="Mask padding (top, right, bottom, left) as percentages",
    )

    # Swap settings
    swap_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Blend between source and target embeddings (0=target, 1=source)",
    )

    # Output settings
    output_format: OutputFormatSchema = Field(
        default=OutputFormatSchema.PNG,
        description="Output image format",
    )
    output_quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="Output quality for JPEG/WebP (1-100)",
    )

    # Execution settings
    execution_providers: list[str] | None = Field(
        default=None,
        description="ONNX execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']). None for auto-detect.",
    )

    # Enhancement settings
    face_enhancer_model: FaceEnhancerModelSchema = Field(
        default=FaceEnhancerModelSchema.NONE,
        description="Face enhancement model (GFPGAN/CodeFormer/RestoreFormer++)",
    )
    face_enhancer_blend: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Blend ratio for enhanced face back into frame",
    )
    face_enhancer_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Model-specific weight (e.g., CodeFormer fidelity)",
    )
    frame_enhancer_model: FrameEnhancerModelSchema = Field(
        default=FrameEnhancerModelSchema.NONE,
        description="Frame enhancement / super-resolution model",
    )
    frame_enhancer_blend: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Blend ratio for enhanced frame back into original",
    )

    # Video options (used when target is a video)
    video: VideoConfigSchema | None = Field(
        default=None,
        description="Video processing options (ffmpeg integration)",
    )


class JobCreate(BaseModel):
    """Job creation request."""

    source_id: str
    target_id: str
    config: SwapConfigSchema | None = Field(
        default=None,
        description="Optional swap configuration. Uses defaults if not provided.",
    )


class JobResponse(BaseModel):
    """Job response."""

    id: str
    status: JobStatus
    progress: float = 0.0
    media_type: MediaTypeSchema = MediaTypeSchema.IMAGE
    frame_count: int | None = None
    fps: float | None = None
    output_url: str | None = None
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class UploadResponse(BaseModel):
    """Upload response."""

    id: str
    filename: str
    size: int
    mime_type: str
    media_type: MediaTypeSchema = MediaTypeSchema.IMAGE
