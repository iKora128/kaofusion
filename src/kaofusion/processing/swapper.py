"""Face swapping using InSwapper model.

Core face swap algorithm that replaces faces in images.
"""

from dataclasses import dataclass
import logging

import cv2
import numpy as np

from kaofusion.processing.helper import paste_back, warp_face
from kaofusion.processing.masker import create_box_mask, create_face_mask
from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.types import (
    Embedding,
    Face,
    FaceMaskType,
    FaceSwapperModel,
    Mask,
    SwapConfig,
    VisionFrame,
    WarpTemplate,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SwapperSpec:
    """Metadata describing a swapper model."""

    model_name: str
    template: WarpTemplate
    size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    type: str  # inswapper | hyperswap


SWAPPER_SPECS: dict[FaceSwapperModel, SwapperSpec] = {
    FaceSwapperModel.INSWAPPER_128: SwapperSpec(
        model_name="inswapper_128",
        template=WarpTemplate.ARCFACE_128,
        size=(128, 128),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        type="inswapper",
    ),
    FaceSwapperModel.INSWAPPER_128_FP16: SwapperSpec(
        model_name="inswapper_128_fp16",
        template=WarpTemplate.ARCFACE_128,
        size=(128, 128),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        type="inswapper",
    ),
    FaceSwapperModel.HYPERSWAP_1A_256: SwapperSpec(
        model_name="hyperswap_1a_256",
        template=WarpTemplate.ARCFACE_128,
        size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        type="hyperswap",
    ),
    FaceSwapperModel.HYPERSWAP_1B_256: SwapperSpec(
        model_name="hyperswap_1b_256",
        template=WarpTemplate.ARCFACE_128,
        size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        type="hyperswap",
    ),
    FaceSwapperModel.HYPERSWAP_1C_256: SwapperSpec(
        model_name="hyperswap_1c_256",
        template=WarpTemplate.ARCFACE_128,
        size=(256, 256),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        type="hyperswap",
    ),
}


class FaceSwapper:
    """Face swapper using InSwapper model."""

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        mask_blur: float = 0.3,
        mask_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        """Initialize face swapper.

        Args:
            model_manager: Model manager instance
            mask_blur: Mask blur amount
            mask_padding: Mask padding (top, right, bottom, left)
        """
        self.model_manager = model_manager or get_model_manager()
        self.mask_blur = mask_blur
        self.mask_padding = mask_padding
        self._model_initializer: np.ndarray | None = None
        self._initializer_model: str | None = None
        self._swapper_spec: SwapperSpec = SWAPPER_SPECS[
            FaceSwapperModel.INSWAPPER_128
        ]

    def swap(
        self,
        source_face: Face,
        target_face: Face,
        target_image: VisionFrame,
        config: SwapConfig,
    ) -> VisionFrame:
        """Swap source face onto target face in image.

        Args:
            source_face: Source face with embedding
            target_face: Target face to replace
            target_image: Target image
            config: Swap configuration

        Returns:
            Image with swapped face
        """
        if source_face.embedding is None:
            raise ValueError("Source face must have embedding")
        if target_face.embedding is None:
            raise ValueError("Target face must have embedding")

        spec = SWAPPER_SPECS.get(config.swapper_model, self._swapper_spec)
        self._swapper_spec = spec

        # Get landmarks for warping
        landmarks_5 = target_face.landmarks.five

        # Warp face to canonical position
        warped_face, affine_matrix = warp_face(
            target_image,
            landmarks_5,
            spec.template,
            spec.size,
        )

        # Create mask (respect occlusion setting)
        use_occlusion = FaceMaskType.OCCLUSION in config.mask_types
        if use_occlusion:
            mask = create_face_mask(
                warped_face,
                use_box=True,
                use_occlusion=True,
                box_blur=self.mask_blur,
                box_padding=self.mask_padding,
                model_manager=self.model_manager,
            )
        else:
            mask = create_box_mask(
                spec.size,
                blur=self.mask_blur,
                padding=self.mask_padding,
            )

        # Prepare source embedding
        source_embedding = self._prepare_embedding(
            source_face,
            target_face,
            spec,
            swap_weight=config.swap_weight,
        )

        # Prepare input frame
        input_frame = self._prepare_input(warped_face, spec)

        # Run swap
        swapped_frame = self._forward(source_embedding, input_frame, spec)

        # Denormalize output
        swapped_face = self._denormalize_output(swapped_frame, spec)

        # Paste back
        logger.debug(f"Swapped face shape: {swapped_face.shape}, min: {swapped_face.min()}, max: {swapped_face.max()}")
        logger.debug(f"Mask shape: {mask.shape}, min: {mask.min()}, max: {mask.max()}")
        logger.debug(f"Affine matrix: {affine_matrix}")

        result = paste_back(target_image, swapped_face, mask, affine_matrix)

        logger.debug(f"Result image shape: {result.shape}")

        return result

    def _prepare_embedding(
        self,
        source_face: Face,
        target_face: Face,
        spec: SwapperSpec,
        swap_weight: float = 0.5,
    ) -> np.ndarray:
        """Prepare source embedding for InSwapper.

        Args:
            source_face: Source face
            target_face: Target face
            spec: Swapper model spec
            swap_weight: Blend between source and target embeddings

        Returns:
            Prepared embedding
        """
        embedding = source_face.embedding.reshape((1, -1))
        target_embedding = (
            target_face.embedding_norm
            if target_face.embedding_norm is not None
            else target_face.embedding
        )

        if spec.type == "hyperswap":
            if source_face.embedding_norm is not None:
                embedding = source_face.embedding_norm.reshape((1, -1))
        elif spec.type == "inswapper":
            # Get model initializer for embedding transformation
            if self._model_initializer is None or self._initializer_model != spec.model_name:
                self._model_initializer = self.model_manager.get_model_initializer(
                    spec.model_name
                )
                self._initializer_model = spec.model_name

            if self._model_initializer is not None:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = np.dot(embedding, self._model_initializer) / norm
            else:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

        # Blend source and target embeddings if requested
        if target_embedding is not None:
            target_norm = np.linalg.norm(target_embedding)
            target_unit = (
                target_embedding / target_norm if target_norm > 0 else target_embedding
            ).reshape(1, -1)
            alpha = float(np.clip(swap_weight, 0.0, 1.0))
            embedding = embedding * alpha + target_unit * (1.0 - alpha)

        return embedding.astype(np.float32)

    def _prepare_input(
        self,
        face_image: VisionFrame,
        spec: SwapperSpec,
    ) -> np.ndarray:
        """Prepare face image for model input.

        Args:
            face_image: Warped face image (BGR)
            spec: Swapper model spec

        Returns:
            Input tensor
        """
        # Convert BGR to RGB and normalize to [0, 1]
        normalized = face_image[:, :, ::-1].astype(np.float32) / 255.0

        # Apply mean and std
        normalized = (normalized - np.array(spec.mean)) / np.array(spec.std)

        # HWC to NCHW
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor.astype(np.float32)

    def _forward(
        self,
        source_embedding: np.ndarray,
        target_frame: np.ndarray,
        spec: SwapperSpec,
    ) -> np.ndarray:
        """Run InSwapper model.

        Args:
            source_embedding: Prepared source embedding
            target_frame: Prepared target frame
            spec: Swapper model spec

        Returns:
            Swapped face tensor
        """
        logger.debug(f"Running swap with model: {spec.model_name}")
        logger.debug(f"Source embedding shape: {source_embedding.shape}, dtype: {source_embedding.dtype}")
        logger.debug(f"Target frame shape: {target_frame.shape}, dtype: {target_frame.dtype}")

        outputs = self.model_manager.run_inference(
            spec.model_name,
            {
                "source": source_embedding,
                "target": target_frame,
            },
        )

        logger.debug(f"Model output shape: {outputs[0].shape}, min: {outputs[0].min()}, max: {outputs[0].max()}")

        return outputs[0][0]

    def _denormalize_output(
        self,
        output: np.ndarray,
        spec: SwapperSpec,
    ) -> VisionFrame:
        """Denormalize model output to image.

        Args:
            output: Model output (CHW format)
            spec: Swapper model spec

        Returns:
            BGR image
        """
        # CHW to HWC
        output = output.transpose(1, 2, 0)

        if spec.type in ("hyperswap",):
            output = output * np.array(spec.std) + np.array(spec.mean)

        # Clip to valid range
        output = np.clip(output, 0, 1)

        # Convert to BGR uint8
        output = (output[:, :, ::-1] * 255).astype(np.uint8)

        return output


def swap_face(
    source_face: Face,
    target_face: Face,
    target_image: VisionFrame,
    model_manager: ModelManager | None = None,
    config: SwapConfig | None = None,
) -> VisionFrame:
    """Convenience function to swap faces.

    Args:
        source_face: Source face with embedding
        target_face: Target face to replace
        target_image: Target image
        model_manager: Optional model manager
        config: Optional swap configuration

    Returns:
        Image with swapped face
    """
    if config is None:
        config = SwapConfig()
    swapper = FaceSwapper(model_manager=model_manager)
    return swapper.swap(source_face, target_face, target_image, config)
