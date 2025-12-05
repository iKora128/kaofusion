"""Face recognition using ArcFace model.

Computes face embeddings for identity comparison and face swapping.
"""

import logging

import numpy as np

from kaofusion.processing.helper import warp_face
from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.types import (
    Embedding,
    Landmarks5,
    VisionFrame,
    WarpTemplate,
)

logger = logging.getLogger(__name__)

# ArcFace model configuration
ARCFACE_TEMPLATE = WarpTemplate.ARCFACE_112_V2
ARCFACE_SIZE = (112, 112)


class FaceRecognizer:
    """Face recognizer using ArcFace model."""

    def __init__(
        self,
        model_manager: ModelManager | None = None,
    ):
        """Initialize face recognizer.

        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager or get_model_manager()

    def compute_embedding(
        self,
        image: VisionFrame,
        landmarks_5: Landmarks5,
    ) -> tuple[Embedding, Embedding]:
        """Compute face embedding from image and landmarks.

        Args:
            image: Input image (BGR)
            landmarks_5: 5-point face landmarks

        Returns:
            Tuple of (raw embedding, normalized embedding)
        """
        # Warp face to ArcFace canonical position
        warped, _ = warp_face(
            image,
            landmarks_5,
            ARCFACE_TEMPLATE,
            ARCFACE_SIZE,
        )

        # Prepare input: normalize to [-1, 1] and convert to NCHW
        normalized = warped.astype(np.float32) / 127.5 - 1
        # BGR to RGB
        normalized = normalized[:, :, ::-1]
        # HWC to NCHW
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Run inference
        outputs = self.model_manager.run_inference(
            "arcface_w600k_r50",
            {"input": input_tensor.astype(np.float32)},
        )

        # Get embedding
        embedding = outputs[0].ravel().astype(np.float32)

        # Compute normalized embedding
        norm = np.linalg.norm(embedding)
        embedding_norm = embedding / norm if norm > 0 else embedding

        return embedding, embedding_norm


def compute_face_embedding(
    image: VisionFrame,
    landmarks_5: Landmarks5,
    model_manager: ModelManager | None = None,
) -> tuple[Embedding, Embedding]:
    """Convenience function to compute face embedding.

    Args:
        image: Input image (BGR)
        landmarks_5: 5-point face landmarks
        model_manager: Optional model manager

    Returns:
        Tuple of (raw embedding, normalized embedding)
    """
    recognizer = FaceRecognizer(model_manager=model_manager)
    return recognizer.compute_embedding(image, landmarks_5)
