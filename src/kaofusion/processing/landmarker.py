"""Face landmark detection using 2DFAN4 model.

Detects 68-point facial landmarks for face alignment and manipulation.
"""

import logging

import cv2
import numpy as np

from kaofusion.processing.helper import transform_points
from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.types import (
    BoundingBox,
    Landmarks,
    Landmarks5,
    Landmarks68,
    VisionFrame,
)

logger = logging.getLogger(__name__)


class FaceLandmarker:
    """Face landmark detector using 2DFAN4 model."""

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        input_size: tuple[int, int] = (256, 256),
    ):
        """Initialize face landmarker.

        Args:
            model_manager: Model manager instance
            input_size: Model input size (width, height)
        """
        self.model_manager = model_manager or get_model_manager()
        self.input_size = input_size

    def detect(
        self,
        image: VisionFrame,
        bbox: BoundingBox,
        landmarks_5: Landmarks5 | None = None,
    ) -> tuple[Landmarks68 | None, float]:
        """Detect 68-point landmarks for a face.

        Args:
            image: Input image (BGR)
            bbox: Face bounding box
            landmarks_5: Optional initial 5-point landmarks

        Returns:
            Tuple of (68-point landmarks, confidence score)
        """
        # Calculate crop parameters
        model_w, model_h = self.input_size
        scale = 195 / max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1)
        translation = (model_w - (bbox[2] + bbox[0]) * scale) * 0.5, \
                      (model_h - (bbox[3] + bbox[1]) * scale) * 0.5

        # Create affine matrix for cropping
        affine_matrix = np.array([
            [scale, 0, translation[0]],
            [0, scale, translation[1]],
        ], dtype=np.float32)

        # Warp face to canonical position
        crop = cv2.warpAffine(image, affine_matrix, self.input_size)

        # Optimize contrast if needed
        crop = self._optimize_contrast(crop)

        # Prepare input
        input_tensor = crop.transpose(2, 0, 1).astype(np.float32) / 255.0

        # Run inference
        try:
            outputs = self.model_manager.run_inference(
                "2dfan4",
                {"input": [input_tensor]},
            )
        except Exception as e:
            logger.error(f"Landmark detection failed: {e}")
            return None, 0.0

        # Parse output
        landmarks_68, heatmap = outputs[0], outputs[1]

        # Convert from heatmap coordinates to image coordinates
        landmarks_68 = landmarks_68[:, :, :2][0] / 64 * 256

        # Transform back to original image space
        inverse_matrix = cv2.invertAffineTransform(affine_matrix)
        landmarks_68 = transform_points(landmarks_68, inverse_matrix)

        # Calculate score from heatmap
        score = float(np.mean(np.amax(heatmap, axis=(2, 3))))
        score = np.interp(score, [0, 0.9], [0, 1])

        return landmarks_68.astype(np.float32), score

    def _optimize_contrast(self, image: VisionFrame) -> VisionFrame:
        """Optimize contrast for dark images.

        Args:
            image: Input image (BGR)

        Returns:
            Contrast-optimized image
        """
        # Convert to Lab color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # Check if image is too dark
        if np.mean(lab[:, :, 0]) < 30:
            # Apply CLAHE to luminance channel
            clahe = cv2.createCLAHE(clipLimit=2.0)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def estimate_landmarks_68_from_5(
    landmarks_5: Landmarks5,
    model_manager: ModelManager | None = None,
) -> Landmarks68:
    """Estimate 68-point landmarks from 5-point landmarks.

    This function is currently not fully implemented as the fan_68_5 model
    is not available. It approximates 68-point landmarks from 5-point
    landmarks using geometric interpolation.

    Note: For best results, use FaceLandmarker.detect() which uses the
    2dfan4 model to directly detect 68-point landmarks.

    Args:
        landmarks_5: 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        model_manager: Model manager instance (unused in current implementation)

    Returns:
        Approximated 68-point landmarks
    """
    # Approximate 68-point landmarks from 5-point landmarks
    # This is a rough approximation for cases where full detection is not available
    left_eye = landmarks_5[0]
    right_eye = landmarks_5[1]
    nose = landmarks_5[2]
    left_mouth = landmarks_5[3]
    right_mouth = landmarks_5[4]

    # Calculate face dimensions
    eye_center = (left_eye + right_eye) / 2
    eye_distance = np.linalg.norm(right_eye - left_eye)
    mouth_center = (left_mouth + right_mouth) / 2

    # Create 68 landmarks by interpolation and extrapolation
    landmarks_68 = np.zeros((68, 2), dtype=np.float32)

    # Approximate face contour (points 0-16)
    face_width = eye_distance * 2.2
    face_top = eye_center[1] - eye_distance * 0.8
    face_bottom = mouth_center[1] + eye_distance * 0.6
    for i in range(17):
        t = i / 16.0
        x = eye_center[0] - face_width / 2 + face_width * t
        y = face_top + (face_bottom - face_top) * (0.5 - abs(t - 0.5))
        landmarks_68[i] = [x, y]

    # Left eyebrow (points 17-21)
    for i in range(5):
        t = i / 4.0
        landmarks_68[17 + i] = left_eye + np.array([
            (-0.3 + 0.6 * t) * eye_distance,
            -0.4 * eye_distance
        ])

    # Right eyebrow (points 22-26)
    for i in range(5):
        t = i / 4.0
        landmarks_68[22 + i] = right_eye + np.array([
            (-0.3 + 0.6 * t) * eye_distance,
            -0.4 * eye_distance
        ])

    # Nose bridge (points 27-30)
    for i in range(4):
        t = i / 3.0
        landmarks_68[27 + i] = eye_center + (nose - eye_center) * t

    # Nose bottom (points 31-35)
    nose_width = eye_distance * 0.5
    landmarks_68[31] = nose + np.array([-nose_width, 0])
    landmarks_68[32] = nose + np.array([-nose_width * 0.5, 0.1 * eye_distance])
    landmarks_68[33] = nose
    landmarks_68[34] = nose + np.array([nose_width * 0.5, 0.1 * eye_distance])
    landmarks_68[35] = nose + np.array([nose_width, 0])

    # Left eye (points 36-41)
    eye_w = eye_distance * 0.35
    eye_h = eye_distance * 0.15
    for i in range(6):
        angle = i * np.pi / 3 + np.pi
        landmarks_68[36 + i] = left_eye + np.array([
            np.cos(angle) * eye_w,
            np.sin(angle) * eye_h
        ])

    # Right eye (points 42-47)
    for i in range(6):
        angle = i * np.pi / 3
        landmarks_68[42 + i] = right_eye + np.array([
            np.cos(angle) * eye_w,
            np.sin(angle) * eye_h
        ])

    # Outer lip (points 48-59)
    mouth_width = np.linalg.norm(right_mouth - left_mouth)
    mouth_height = mouth_width * 0.4
    for i in range(12):
        angle = i * np.pi / 6 + np.pi
        landmarks_68[48 + i] = mouth_center + np.array([
            np.cos(angle) * mouth_width / 2,
            np.sin(angle) * mouth_height / 2
        ])

    # Inner lip (points 60-67)
    inner_width = mouth_width * 0.6
    inner_height = mouth_height * 0.5
    for i in range(8):
        angle = i * np.pi / 4 + np.pi
        landmarks_68[60 + i] = mouth_center + np.array([
            np.cos(angle) * inner_width / 2,
            np.sin(angle) * inner_height / 2
        ])

    return landmarks_68


def detect_landmarks(
    image: VisionFrame,
    bbox: BoundingBox,
    model_manager: ModelManager | None = None,
) -> Landmarks | None:
    """Convenience function to detect landmarks.

    Args:
        image: Input image (BGR)
        bbox: Face bounding box
        model_manager: Optional model manager

    Returns:
        Landmarks object or None if detection failed
    """
    landmarker = FaceLandmarker(model_manager=model_manager)
    landmarks_68, score = landmarker.detect(image, bbox)

    if landmarks_68 is None or score < 0.3:
        return None

    # Derive 5-point landmarks from 68-point
    landmarks_5 = np.array([
        np.mean(landmarks_68[36:42], axis=0),  # Left eye
        np.mean(landmarks_68[42:48], axis=0),  # Right eye
        landmarks_68[30],  # Nose tip
        landmarks_68[48],  # Left mouth corner
        landmarks_68[54],  # Right mouth corner
    ], dtype=np.float32)

    return Landmarks(
        landmarks_5=landmarks_5,
        landmarks_68=landmarks_68,
    )
