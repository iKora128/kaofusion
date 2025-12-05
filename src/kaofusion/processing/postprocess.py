"""Post-processing pipeline (face/frame enhancers)."""

from dataclasses import dataclass
import logging
from typing import Sequence

import cv2
import numpy as np

from kaofusion.processing.helper import paste_back, warp_face
from kaofusion.processing.masker import create_face_mask
from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.types import (
    Face,
    FaceEnhancerModel,
    FaceMaskType,
    FrameEnhancerModel,
    SwapConfig,
    VisionFrame,
    WarpTemplate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceEnhancerSpec:
    """Metadata for a face enhancer."""

    model_name: str
    template: WarpTemplate
    size: tuple[int, int]


@dataclass(frozen=True)
class FrameEnhancerSpec:
    """Metadata for a frame enhancer."""

    model_name: str
    scale: int


# Supported face enhancers
FACE_ENHANCER_SPECS: dict[FaceEnhancerModel, FaceEnhancerSpec] = {
    FaceEnhancerModel.GFPGAN_1_4: FaceEnhancerSpec(
        model_name="gfpgan_1.4",
        template=WarpTemplate.FFHQ_512,
        size=(512, 512),
    ),
    FaceEnhancerModel.CODEFORMER: FaceEnhancerSpec(
        model_name="codeformer",
        template=WarpTemplate.FFHQ_512,
        size=(512, 512),
    ),
    FaceEnhancerModel.RESTOREFORMER_PLUS_PLUS: FaceEnhancerSpec(
        model_name="restoreformer_plus_plus",
        template=WarpTemplate.FFHQ_512,
        size=(512, 512),
    ),
}


# Supported frame enhancers
FRAME_ENHANCER_SPECS: dict[FrameEnhancerModel, FrameEnhancerSpec] = {
    FrameEnhancerModel.REAL_ESRGAN_X4: FrameEnhancerSpec(
        model_name="real_esrgan_x4",
        scale=4,
    ),
}


class FaceEnhancer:
    """Wrap GFPGAN/CodeFormer/RestoreFormer++ style enhancers."""

    def __init__(self, model_manager: ModelManager | None = None):
        self.model_manager = model_manager or get_model_manager()
        self._input_cache: dict[str, set[str]] = {}

    def enhance(
        self,
        image: VisionFrame,
        face: Face,
        config: SwapConfig,
    ) -> VisionFrame:
        """Enhance a single face region."""
        spec = FACE_ENHANCER_SPECS.get(config.face_enhancer_model)
        if spec is None:
            return image

        # Warp face to model space
        crop, matrix = warp_face(
            image,
            face.landmarks.five,
            spec.template,
            spec.size,
        )

        # Build mask (respect occlusion if configured)
        use_occlusion = FaceMaskType.OCCLUSION in config.mask_types
        mask = create_face_mask(
            crop,
            use_box=True,
            use_occlusion=use_occlusion,
            box_blur=config.mask_blur,
            box_padding=config.mask_padding,
            model_manager=self.model_manager,
        )

        # Prepare model input
        prepared = self._prepare_input(crop)
        inputs = {"input": prepared}

        input_names = self._get_input_names(spec.model_name)
        if "weight" in input_names:
            weight = np.array([config.face_enhancer_weight], dtype=np.float32)
            inputs["weight"] = weight

        try:
            outputs = self.model_manager.run_inference(spec.model_name, inputs)
        except Exception:
            logger.exception("Face enhancer inference failed, skipping enhancement")
            return image

        enhanced = self._normalize_output(outputs[0][0])
        pasted = paste_back(image, enhanced, mask, matrix)

        blend = float(np.clip(config.face_enhancer_blend, 0.0, 1.0))
        blended = (
            image.astype(np.float32) * (1.0 - blend)
            + pasted.astype(np.float32) * blend
        )
        return blended.astype(np.uint8)

    def _prepare_input(self, crop: VisionFrame) -> np.ndarray:
        """Normalize input for enhancer models."""
        normalized = crop[:, :, ::-1].astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5  # [-1, 1]
        normalized = normalized.transpose(2, 0, 1)
        return np.expand_dims(normalized, axis=0).astype(np.float32)

    def _normalize_output(self, output: np.ndarray) -> VisionFrame:
        """Convert model output back to BGR image."""
        output = np.clip(output, -1, 1)
        output = (output + 1.0) / 2.0
        output = output.transpose(1, 2, 0)
        output = np.clip(output * 255.0, 0, 255)
        return output.astype(np.uint8)[:, :, ::-1]

    def _get_input_names(self, model_name: str) -> set[str]:
        """Cache input names to avoid querying sessions repeatedly."""
        if model_name not in self._input_cache:
            try:
                session = self.model_manager.get_session(model_name)
                self._input_cache[model_name] = {
                    inp.name for inp in session.get_inputs()
                }
            except Exception:
                self._input_cache[model_name] = {"input"}
        return self._input_cache[model_name]


class FrameEnhancer:
    """Wrap frame enhancers such as Real-ESRGAN."""

    def __init__(self, model_manager: ModelManager | None = None):
        self.model_manager = model_manager or get_model_manager()

    def enhance(
        self,
        frame: VisionFrame,
        config: SwapConfig,
    ) -> VisionFrame:
        """Enhance a full frame."""
        spec = FRAME_ENHANCER_SPECS.get(config.frame_enhancer_model)
        if spec is None:
            return frame

        prepared = self._prepare_input(frame)

        try:
            outputs = self.model_manager.run_inference(
                spec.model_name,
                {"input": prepared},
            )
        except Exception:
            logger.exception("Frame enhancer inference failed, returning original frame")
            return frame

        enhanced = self._normalize_output(outputs[0][0])
        blend = float(np.clip(config.frame_enhancer_blend, 0.0, 1.0))

        if enhanced.shape[0:2] != frame.shape[0:2]:
            base = cv2.resize(frame, (enhanced.shape[1], enhanced.shape[0]))
        else:
            base = frame

        blended = base.astype(np.float32) * (1.0 - blend) + enhanced.astype(
            np.float32
        ) * blend
        return blended.astype(np.uint8)

    def _prepare_input(self, frame: VisionFrame) -> np.ndarray:
        """Prepare frame tensor for ESRGAN-like models."""
        normalized = frame[:, :, ::-1].astype(np.float32) / 255.0
        normalized = normalized.transpose(2, 0, 1)
        return np.expand_dims(normalized, axis=0).astype(np.float32)

    def _normalize_output(self, output: np.ndarray) -> VisionFrame:
        """Convert ESRGAN output back to BGR uint8."""
        output = np.clip(output, 0.0, 1.0)
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).astype(np.uint8)
        return output[:, :, ::-1]


class PostProcessorChain:
    """Ordered post-processing pipeline."""

    def __init__(
        self,
        face_enhancer: FaceEnhancer | None = None,
        frame_enhancer: FrameEnhancer | None = None,
    ):
        self.face_enhancer = face_enhancer
        self.frame_enhancer = frame_enhancer

    def apply(
        self,
        image: VisionFrame,
        swapped_faces: Sequence[Face],
        config: SwapConfig,
    ) -> VisionFrame:
        """Apply configured post-processors."""
        result = image

        if (
            self.face_enhancer
            and config.face_enhancer_model != FaceEnhancerModel.NONE
            and swapped_faces
        ):
            for face in swapped_faces:
                if face.landmarks:
                    result = self.face_enhancer.enhance(result, face, config)

        if self.frame_enhancer and config.frame_enhancer_model != FrameEnhancerModel.NONE:
            result = self.frame_enhancer.enhance(result, config)

        return result
