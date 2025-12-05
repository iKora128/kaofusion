"""Image I/O and manipulation utilities.

Clean implementation of image reading, writing, and transformation functions.
"""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from kaofusion.processing.types import Mask, VisionFrame


def read_image(path: str | Path, color_mode: Literal["bgr", "rgb", "rgba"] = "bgr") -> VisionFrame | None:
    """Read an image from disk.

    Args:
        path: Path to image file
        color_mode: Color mode - 'bgr' (default OpenCV), 'rgb', or 'rgba'

    Returns:
        Image as numpy array or None if failed
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None

    flag = cv2.IMREAD_COLOR
    if color_mode == "rgba":
        flag = cv2.IMREAD_UNCHANGED

    # Use numpy fromfile for unicode path support (Windows compatibility)
    image_buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_buffer, flag)

    if image is None:
        return None

    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def write_image(path: str | Path, image: VisionFrame, quality: int | None = None) -> bool:
    """Write an image to disk.

    Args:
        path: Output path
        image: Image to write (BGR format)
        quality: JPEG/WebP quality (1-100). None for format defaults.

    Returns:
        True if successful
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    # Set encoding parameters based on format
    params: list[int] = []
    if ext in [".jpg", ".jpeg"]:
        jpeg_quality = quality if quality is not None else 95
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif ext == ".webp":
        webp_quality = quality if quality is not None else 95
        params = [cv2.IMWRITE_WEBP_QUALITY, webp_quality]
    elif ext == ".png":
        # PNG compression level 0-9 (quality doesn't apply, always use best compression)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]

    # Encode and write using numpy for unicode path support
    success, encoded = cv2.imencode(ext, image, params)
    if not success:
        return False

    encoded.tofile(str(path))
    return path.exists()


def resize_image(
    image: VisionFrame,
    size: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> VisionFrame:
    """Resize an image.

    Args:
        image: Input image
        size: Target (width, height)
        interpolation: OpenCV interpolation method

    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=interpolation)


def get_image_size(image: VisionFrame) -> tuple[int, int]:
    """Get image dimensions.

    Args:
        image: Input image

    Returns:
        (width, height)
    """
    h, w = image.shape[:2]
    return w, h


def fit_to_size(
    image: VisionFrame,
    target_size: tuple[int, int],
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> VisionFrame:
    """Fit image to target size while maintaining aspect ratio.

    The image is scaled to fit within target_size and centered with padding.

    Args:
        image: Input image
        target_size: Target (width, height)
        pad_color: Padding color (BGR)

    Returns:
        Fitted image
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # Calculate scale to fit
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate padding
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # Pad
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return padded


def crop_image(
    image: VisionFrame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> VisionFrame:
    """Crop a region from an image.

    Args:
        image: Input image
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner

    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    return image[y1:y2, x1:x2].copy()


def blend_images(
    source: VisionFrame,
    target: VisionFrame,
    mask: Mask,
) -> VisionFrame:
    """Blend source onto target using mask.

    Args:
        source: Source image
        target: Target image
        mask: Blend mask (0-1 float, same size as images)

    Returns:
        Blended image
    """
    # Ensure mask is 3-channel for broadcasting
    if mask.ndim == 2:
        mask = mask[:, :, np.newaxis]

    # Blend
    blended = source * mask + target * (1 - mask)
    return blended.astype(np.uint8)


def apply_affine_transform(
    image: VisionFrame,
    matrix: np.ndarray,
    output_size: tuple[int, int],
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> VisionFrame:
    """Apply affine transformation to image.

    Args:
        image: Input image
        matrix: 2x3 affine transformation matrix
        output_size: Output (width, height)
        border_value: Border fill color (BGR)

    Returns:
        Transformed image
    """
    return cv2.warpAffine(
        image,
        matrix,
        output_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def create_gaussian_blur_mask(
    size: tuple[int, int],
    blur_amount: float = 0.3,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Mask:
    """Create a gaussian-blurred box mask.

    Args:
        size: Mask (width, height)
        blur_amount: Blur factor (0-1)
        padding: (top, right, bottom, left) padding

    Returns:
        Mask with blurred edges
    """
    w, h = size
    top, right, bottom, left = padding

    # Create base mask
    mask = np.ones((h, w), dtype=np.float32)

    # Apply padding (set edges to 0)
    if top > 0:
        mask[:top, :] = 0
    if bottom > 0:
        mask[-bottom:, :] = 0
    if left > 0:
        mask[:, :left] = 0
    if right > 0:
        mask[:, -right:] = 0

    # Apply gaussian blur
    if blur_amount > 0:
        blur_size = int(min(w, h) * blur_amount)
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size > 1:
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask


def normalize_for_model(
    image: VisionFrame,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Normalize image for model input.

    Args:
        image: Input image (BGR, uint8)
        mean: Per-channel mean
        std: Per-channel standard deviation

    Returns:
        Normalized float32 array in NCHW format
    """
    # Convert to float and normalize to 0-1
    normalized = image[:, :, ::-1].astype(np.float32) / 255.0

    # Apply mean and std
    normalized = (normalized - np.array(mean)) / np.array(std)

    # Convert to NCHW format
    normalized = normalized.transpose(2, 0, 1)
    normalized = np.expand_dims(normalized, axis=0)

    return normalized.astype(np.float32)


def denormalize_from_model(
    output: np.ndarray,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> VisionFrame:
    """Denormalize model output to image.

    Args:
        output: Model output in CHW format
        mean: Per-channel mean used in normalization
        std: Per-channel standard deviation used in normalization

    Returns:
        BGR uint8 image
    """
    # Remove batch dimension if present
    if output.ndim == 4:
        output = output[0]

    # Convert from CHW to HWC
    output = output.transpose(1, 2, 0)

    # Reverse normalization
    output = output * np.array(std) + np.array(mean)

    # Clip and convert
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    # Convert RGB to BGR
    output = output[:, :, ::-1]

    return output
