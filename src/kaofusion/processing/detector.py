"""Face detection using RetinaFace model.

Clean implementation of face detection with ONNX Runtime.
"""

import logging

import cv2
import numpy as np

from kaofusion.processing.helper import (
    apply_nms,
    create_anchors,
    distance_to_bbox,
    distance_to_landmarks5,
    normalize_bbox,
)
from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.types import (
    BoundingBox,
    Face,
    FaceScores,
    Landmarks,
    Landmarks5,
    VisionFrame,
)

logger = logging.getLogger(__name__)

# RetinaFace model configuration
FEATURE_STRIDES = [8, 16, 32]
FEATURE_MAP_CHANNEL = 3
ANCHOR_COUNT = 2


class FaceDetector:
    """Face detector using RetinaFace model."""

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
    ):
        """Initialize face detector.

        Args:
            model_manager: Model manager instance
            score_threshold: Minimum detection score
            nms_threshold: NMS IoU threshold
            input_size: Detection input size (width, height)
        """
        self.model_manager = model_manager or get_model_manager()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

    def detect(self, image: VisionFrame) -> list[Face]:
        """Detect faces in an image.

        Args:
            image: Input image (BGR)

        Returns:
            List of detected Face objects
        """
        # Prepare image for detection
        orig_h, orig_w = image.shape[:2]
        det_w, det_h = self.input_size

        # Resize maintaining aspect ratio
        scale = min(det_w / orig_w, det_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to detection size
        padded = np.zeros((det_h, det_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize and prepare input tensor
        input_tensor = self._prepare_input(padded)

        # Run inference
        outputs = self._run_inference(input_tensor)

        # Parse detections
        bboxes, scores, landmarks = self._parse_outputs(
            outputs, det_w, det_h, scale
        )

        # Apply NMS
        keep_indices = apply_nms(
            bboxes, scores, self.score_threshold, self.nms_threshold
        )

        # Create Face objects
        faces = []
        for idx in keep_indices:
            face = Face(
                bbox=bboxes[idx],
                landmarks=Landmarks(landmarks_5=landmarks[idx]),
                scores=FaceScores(detector=scores[idx]),
            )
            faces.append(face)

        return faces

    def _prepare_input(self, image: VisionFrame) -> np.ndarray:
        """Prepare image for model input.

        Args:
            image: Padded image at detection size

        Returns:
            Input tensor (1, 3, H, W)
        """
        # Normalize to [-1, 1]
        normalized = (image.astype(np.float32) - 127.5) / 128.0

        # Convert HWC to NCHW
        transposed = normalized.transpose(2, 0, 1)
        batched = np.expand_dims(transposed, axis=0)

        return batched.astype(np.float32)

    def _run_inference(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run model inference.

        Args:
            input_tensor: Prepared input tensor

        Returns:
            Model outputs
        """
        return self.model_manager.run_inference(
            "retinaface_10g",
            {"input": input_tensor},
        )

    def _parse_outputs(
        self,
        outputs: list[np.ndarray],
        det_w: int,
        det_h: int,
        scale: float,
    ) -> tuple[list[BoundingBox], list[float], list[Landmarks5]]:
        """Parse model outputs to detections.

        Args:
            outputs: Model output tensors
            det_w: Detection width
            det_h: Detection height
            scale: Scale factor from original image

        Returns:
            Tuple of (bboxes, scores, landmarks)
        """
        all_bboxes: list[BoundingBox] = []
        all_scores: list[float] = []
        all_landmarks: list[Landmarks5] = []

        for idx, stride in enumerate(FEATURE_STRIDES):
            # Get outputs for this scale
            scores_raw = outputs[idx]
            bboxes_raw = outputs[idx + FEATURE_MAP_CHANNEL] * stride
            landmarks_raw = outputs[idx + FEATURE_MAP_CHANNEL * 2] * stride

            # Find detections above threshold
            keep_mask = scores_raw >= self.score_threshold
            keep_indices = np.where(keep_mask)[0]

            if len(keep_indices) == 0:
                continue

            # Create anchors for this scale
            stride_h = det_h // stride
            stride_w = det_w // stride
            anchors = create_anchors(stride, ANCHOR_COUNT, stride_h, stride_w)

            # Convert distances to bboxes
            bboxes = distance_to_bbox(anchors, bboxes_raw)
            landmarks = distance_to_landmarks5(anchors, landmarks_raw)

            # Filter and scale
            for i in keep_indices:
                bbox = bboxes[i] / scale
                bbox = normalize_bbox(bbox)
                all_bboxes.append(bbox)

                all_scores.append(float(scores_raw[i, 0]))

                lm = landmarks[i] / scale
                all_landmarks.append(lm.astype(np.float32))

        return all_bboxes, all_scores, all_landmarks


def detect_faces(
    image: VisionFrame,
    score_threshold: float = 0.5,
    model_manager: ModelManager | None = None,
) -> list[Face]:
    """Convenience function to detect faces.

    Args:
        image: Input image (BGR)
        score_threshold: Minimum detection score
        model_manager: Optional model manager

    Returns:
        List of detected faces
    """
    detector = FaceDetector(
        model_manager=model_manager,
        score_threshold=score_threshold,
    )
    return detector.detect(image)
