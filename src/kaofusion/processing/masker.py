"""Face masking for blending.

Creates masks for seamless face blending during swap.
"""

import logging

import cv2
import numpy as np

from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.types import Mask, VisionFrame

logger = logging.getLogger(__name__)


def create_box_mask(
    size: tuple[int, int],
    blur: float = 0.3,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Mask:
    """Create a box mask with blurred edges.

    Args:
        size: Mask size (width, height)
        blur: Blur amount (0-1)
        padding: Padding percentages (top, right, bottom, left)

    Returns:
        Box mask with blurred edges
    """
    w, h = size
    blur_amount = int(w * 0.5 * blur)
    blur_area = max(blur_amount // 2, 1)

    # Start with all ones
    mask = np.ones((h, w), dtype=np.float32)

    # Apply padding
    top = max(blur_area, int(h * padding[0] / 100))
    right = max(blur_area, int(w * padding[1] / 100))
    bottom = max(blur_area, int(h * padding[2] / 100))
    left = max(blur_area, int(w * padding[3] / 100))

    if top > 0:
        mask[:top, :] = 0
    if bottom > 0:
        mask[-bottom:, :] = 0
    if left > 0:
        mask[:, :left] = 0
    if right > 0:
        mask[:, -right:] = 0

    # Apply blur
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur_amount * 0.25)

    return mask


def create_occlusion_mask(
    face_image: VisionFrame,
    model_manager: ModelManager | None = None,
) -> Mask:
    """Create occlusion mask using XSeg model.

    Detects occlusions like hands, glasses, hair, etc.

    Args:
        face_image: Cropped face image
        model_manager: Model manager instance

    Returns:
        Occlusion mask (0 = occluded, 1 = visible)
    """
    manager = model_manager or get_model_manager()
    model_size = (256, 256)

    # Prepare input
    resized = cv2.resize(face_image, model_size)
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Run inference
    try:
        outputs = manager.run_inference(
            "xseg_1",
            {"input": input_tensor},
        )
        mask = outputs[0][0]
    except Exception as e:
        logger.warning(f"Occlusion detection failed: {e}, using box mask")
        return np.ones(face_image.shape[:2], dtype=np.float32)

    # Process mask
    mask = mask.transpose(0, 1, 2) if mask.ndim == 3 else mask
    mask = np.clip(mask, 0, 1).astype(np.float32)

    # Resize to face image size
    h, w = face_image.shape[:2]
    mask = cv2.resize(mask, (w, h))

    # Smooth and threshold
    mask = cv2.GaussianBlur(mask, (0, 0), 5)
    mask = np.clip((mask - 0.5) * 2 + 0.5, 0, 1)

    return mask


def combine_masks(*masks: Mask) -> Mask:
    """Combine multiple masks using minimum operation.

    Args:
        *masks: Masks to combine

    Returns:
        Combined mask
    """
    if not masks:
        raise ValueError("At least one mask required")

    result = masks[0].copy()
    for mask in masks[1:]:
        result = np.minimum(result, mask)

    return result


def create_face_mask(
    face_image: VisionFrame,
    use_box: bool = True,
    use_occlusion: bool = True,
    box_blur: float = 0.3,
    box_padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    model_manager: ModelManager | None = None,
) -> Mask:
    """Create a combined face mask.

    Args:
        face_image: Cropped face image
        use_box: Whether to use box mask
        use_occlusion: Whether to use occlusion detection
        box_blur: Box mask blur amount
        box_padding: Box mask padding
        model_manager: Model manager instance

    Returns:
        Combined face mask
    """
    h, w = face_image.shape[:2]
    masks = []

    if use_box:
        box_mask = create_box_mask((w, h), box_blur, box_padding)
        masks.append(box_mask)

    if use_occlusion:
        try:
            occlusion_mask = create_occlusion_mask(face_image, model_manager)
            masks.append(occlusion_mask)
        except Exception as e:
            logger.warning(f"Occlusion mask failed: {e}")

    if not masks:
        return np.ones((h, w), dtype=np.float32)

    return combine_masks(*masks)
