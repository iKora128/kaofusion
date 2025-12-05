"""Face helper functions for warping, alignment, and transformation.

Clean implementation of face geometric transformations.
"""

import logging
from functools import lru_cache

import cv2
import numpy as np

from kaofusion.processing.types import (
    AffineMatrix,
    BoundingBox,
    Landmarks5,
    Landmarks68,
    Mask,
    VisionFrame,
    WarpTemplate,
    WARP_TEMPLATES,
)

logger = logging.getLogger(__name__)


def estimate_affine_matrix(
    landmarks: Landmarks5,
    template: WarpTemplate,
    output_size: tuple[int, int],
) -> AffineMatrix:
    """Estimate affine transformation matrix from landmarks to template.

    Args:
        landmarks: 5-point face landmarks (5, 2)
        template: Target warp template
        output_size: Output (width, height)

    Returns:
        2x3 affine transformation matrix
    """
    # Get template points - these are already in pixel coordinates for a specific size
    # ARCFACE_112 templates are for 112x112, ARCFACE_128 for 128x128, FFHQ_512 for 512x512
    template_points = WARP_TEMPLATES[template].copy()

    # Detect native size from template name and scale if needed
    if "112" in template.value:
        native_size = 112
    elif "128" in template.value:
        native_size = 128
    elif "512" in template.value:
        native_size = 512
    else:
        native_size = 128  # default

    # Scale template to match output_size if it differs from native size
    if output_size[0] != native_size or output_size[1] != native_size:
        scale_x = output_size[0] / native_size
        scale_y = output_size[1] / native_size
        template_points = template_points * np.array([[scale_x, scale_y]])

    # Estimate partial affine (rotation, scale, translation)
    # This transforms from source landmarks to template positions
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        template_points.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )

    return matrix.astype(np.float32)


def warp_face(
    image: VisionFrame,
    landmarks: Landmarks5,
    template: WarpTemplate,
    output_size: tuple[int, int],
) -> tuple[VisionFrame, AffineMatrix]:
    """Warp face region to canonical position.

    Args:
        image: Input image
        landmarks: 5-point face landmarks
        template: Target warp template
        output_size: Output (width, height)

    Returns:
        Tuple of (warped face image, affine matrix)
    """
    matrix = estimate_affine_matrix(landmarks, template, output_size)
    warped = cv2.warpAffine(
        image,
        matrix,
        output_size,
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_AREA,
    )
    return warped, matrix


def paste_back(
    target: VisionFrame,
    face: VisionFrame,
    mask: Mask,
    matrix: AffineMatrix,
) -> VisionFrame:
    """Paste warped face back to target image.

    Args:
        target: Target image to paste onto
        face: Warped face image
        mask: Face mask (0-1 float)
        matrix: Affine matrix used for warping

    Returns:
        Target image with face pasted back
    """
    # Calculate paste region
    paste_bbox, paste_matrix = _calculate_paste_region(target, face, matrix)
    x1, y1, x2, y2 = paste_bbox

    paste_width = x2 - x1
    paste_height = y2 - y1

    logger.debug(f"Paste region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    logger.debug(f"Paste size: {paste_width}x{paste_height}")

    if paste_width <= 0 or paste_height <= 0:
        logger.warning("Paste region has zero size, returning original target")
        return target

    # Inverse warp the mask
    inverse_mask = cv2.warpAffine(
        mask,
        paste_matrix,
        (paste_width, paste_height),
    )
    inverse_mask = np.clip(inverse_mask, 0, 1)
    inverse_mask = inverse_mask[:, :, np.newaxis]

    # Inverse warp the face
    inverse_face = cv2.warpAffine(
        face,
        paste_matrix,
        (paste_width, paste_height),
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Blend
    result = target.copy()
    roi = result[y1:y2, x1:x2]

    logger.debug(f"ROI shape: {roi.shape}")
    logger.debug(f"Inverse mask shape: {inverse_mask.shape}, min: {inverse_mask.min()}, max: {inverse_mask.max()}")
    logger.debug(f"Inverse face shape: {inverse_face.shape}, min: {inverse_face.min()}, max: {inverse_face.max()}")

    blended = roi * (1 - inverse_mask) + inverse_face * inverse_mask
    result[y1:y2, x1:x2] = blended.astype(target.dtype)

    logger.debug(f"Blended region applied to result")

    return result


def _calculate_paste_region(
    target: VisionFrame,
    face: VisionFrame,
    matrix: AffineMatrix,
) -> tuple[tuple[int, int, int, int], AffineMatrix]:
    """Calculate the region for pasting face back.

    Args:
        target: Target image
        face: Warped face image
        matrix: Original warp matrix

    Returns:
        Tuple of (bounding box, adjusted inverse matrix)
    """
    target_h, target_w = target.shape[:2]
    face_h, face_w = face.shape[:2]

    # Invert the affine matrix
    inverse_matrix = cv2.invertAffineTransform(matrix)

    # Get corners of face image
    corners = np.array([
        [0, 0],
        [face_w, 0],
        [face_w, face_h],
        [0, face_h],
    ], dtype=np.float32)

    # Transform corners to target space
    transformed = transform_points(corners, inverse_matrix)

    # Get bounding box
    min_pt = np.floor(transformed.min(axis=0)).astype(int)
    max_pt = np.ceil(transformed.max(axis=0)).astype(int)

    # Clip to target bounds
    x1 = max(0, min(min_pt[0], target_w))
    y1 = max(0, min(min_pt[1], target_h))
    x2 = max(0, min(max_pt[0], target_w))
    y2 = max(0, min(max_pt[1], target_h))

    # Adjust inverse matrix for the paste region offset
    paste_matrix = inverse_matrix.copy()
    paste_matrix[0, 2] -= x1
    paste_matrix[1, 2] -= y1

    return (x1, y1, x2, y2), paste_matrix


def transform_points(
    points: np.ndarray,
    matrix: AffineMatrix,
) -> np.ndarray:
    """Apply affine transformation to points.

    Args:
        points: Points array (N, 2)
        matrix: 2x3 affine matrix

    Returns:
        Transformed points (N, 2)
    """
    points = points.reshape(-1, 1, 2)
    transformed = cv2.transform(points, matrix)
    return transformed.reshape(-1, 2)


def transform_bbox(
    bbox: BoundingBox,
    matrix: AffineMatrix,
) -> BoundingBox:
    """Apply affine transformation to bounding box.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        matrix: 2x3 affine matrix

    Returns:
        Transformed bounding box
    """
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[2], bbox[1]],
        [bbox[2], bbox[3]],
        [bbox[0], bbox[3]],
    ], dtype=np.float32)

    transformed = transform_points(points, matrix)
    x1, y1 = transformed.min(axis=0)
    x2, y2 = transformed.max(axis=0)

    return normalize_bbox(np.array([x1, y1, x2, y2], dtype=np.float32))


def normalize_bbox(bbox: BoundingBox) -> BoundingBox:
    """Normalize bounding box to ensure x1 < x2 and y1 < y2.

    Args:
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Normalized bounding box
    """
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return np.array([x1, y1, x2, y2], dtype=np.float32)


@lru_cache(maxsize=16)
def create_anchors(
    stride: int,
    anchor_count: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Create anchor points for detection grid.

    Args:
        stride: Feature stride
        anchor_count: Number of anchors per location
        height: Grid height
        width: Grid width

    Returns:
        Anchors array (N, 2)
    """
    y, x = np.mgrid[:height, :width]
    anchors = np.stack((x, y), axis=-1)
    anchors = (anchors * stride).reshape((-1, 2))
    anchors = np.stack([anchors] * anchor_count, axis=1).reshape((-1, 2))
    return anchors.astype(np.float32)


def distance_to_bbox(
    anchors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Convert anchor distances to bounding boxes.

    Args:
        anchors: Anchor points (N, 2)
        distances: Distances from anchors (N, 4) - left, top, right, bottom

    Returns:
        Bounding boxes (N, 4)
    """
    x1 = anchors[:, 0] - distances[:, 0]
    y1 = anchors[:, 1] - distances[:, 1]
    x2 = anchors[:, 0] + distances[:, 2]
    y2 = anchors[:, 1] + distances[:, 3]
    return np.column_stack([x1, y1, x2, y2])


def distance_to_landmarks5(
    anchors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Convert anchor distances to 5-point landmarks.

    Args:
        anchors: Anchor points (N, 2)
        distances: Distances from anchors (N, 10) - 5 points x 2 coords

    Returns:
        Landmarks (N, 5, 2)
    """
    x = anchors[:, 0::2] + distances[:, 0::2]
    y = anchors[:, 1::2] + distances[:, 1::2]
    landmarks = np.stack((x, y), axis=-1)
    return landmarks


def convert_landmarks68_to_5(landmarks68: Landmarks68) -> Landmarks5:
    """Convert 68-point landmarks to 5-point landmarks.

    Args:
        landmarks68: 68-point landmarks

    Returns:
        5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
    """
    return np.array([
        np.mean(landmarks68[36:42], axis=0),  # Left eye center
        np.mean(landmarks68[42:48], axis=0),  # Right eye center
        landmarks68[30],  # Nose tip
        landmarks68[48],  # Left mouth corner
        landmarks68[54],  # Right mouth corner
    ], dtype=np.float32)


def apply_nms(
    bboxes: list[np.ndarray],
    scores: list[float],
    score_threshold: float,
    nms_threshold: float,
) -> list[int]:
    """Apply non-maximum suppression.

    Args:
        bboxes: List of bounding boxes
        scores: Confidence scores
        score_threshold: Minimum score threshold
        nms_threshold: NMS IoU threshold

    Returns:
        Indices of kept boxes
    """
    if not bboxes:
        return []

    # Convert to xywh format for OpenCV NMS
    bboxes_xywh = [(x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes]

    indices = cv2.dnn.NMSBoxes(
        bboxes_xywh,
        scores,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )

    return list(indices.flatten()) if len(indices) > 0 else []
