"""File upload routes."""

import base64
import io
import uuid
from pathlib import Path

import cv2
import filetype
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from kaofusion.api.schemas.job import MediaTypeSchema, UploadResponse
from kaofusion.config import settings
from kaofusion.processing.detector import FaceDetector
from kaofusion.processing.models import get_model_manager
from kaofusion.processing.vision import read_image
from kaofusion.services.file_service import get_file_service

router = APIRouter(prefix="/api", tags=["uploads"])

# Allowed image MIME types
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/x-msvideo",
}


def get_upload_path(upload_id: str) -> Path | None:
    """Get path for an upload ID."""
    service = get_file_service()
    return service.get_file_path(upload_id)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload an image file.

    Args:
        file: Image file to upload

    Returns:
        Upload info including ID
    """
    # Read file content first
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="File too large")

    # Validate file type using magic bytes (more reliable than Content-Type)
    detected = filetype.guess(content)
    if not detected:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    media_type = None
    if detected.mime in ALLOWED_IMAGE_TYPES:
        media_type = "image"
    elif detected.mime in ALLOWED_VIDEO_TYPES:
        media_type = "video"

    if media_type is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File must be an image or video. "
                f"Supported images: jpeg/png/webp/gif/bmp. "
                f"Supported video: mp4/mov/mkv/avi. "
                f"Got: {detected.mime}"
            ),
        )

    # Generate unique ID
    upload_id = uuid.uuid4().hex[:12]

    # Use detected extension (more reliable)
    ext = f".{detected.extension}"
    filename = f"{upload_id}{ext}"
    file_path = settings.upload_dir / filename

    file_path.write_bytes(content)

    # Register with file service for persistent tracking
    service = get_file_service()
    service.register_file(
        file_id=upload_id,
        filename=filename,
        path=file_path,
        size=len(content),
        mime_type=detected.mime,
    )

    media_type_enum = MediaTypeSchema.VIDEO if media_type == "video" else MediaTypeSchema.IMAGE

    return UploadResponse(
        id=upload_id,
        filename=filename,
        size=len(content),
        mime_type=detected.mime,
        media_type=media_type_enum,
    )


@router.get("/uploads/{upload_id}")
async def get_upload_info(upload_id: str):
    """Get upload info."""
    service = get_file_service()
    record = service.get_file(upload_id)

    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")

    path = Path(record.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload file not found")

    return {
        "id": record.id,
        "filename": record.filename,
        "size": record.size,
        "mime_type": record.mime_type,
        "media_type": MediaTypeSchema.VIDEO if record.mime_type.startswith("video/") else MediaTypeSchema.IMAGE,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
    }


@router.delete("/uploads/{upload_id}")
async def delete_upload(upload_id: str):
    """Delete an upload."""
    service = get_file_service()

    if not service.get_file(upload_id):
        raise HTTPException(status_code=404, detail="Upload not found")

    if service.delete_file(upload_id):
        return {"message": "Upload deleted successfully"}

    raise HTTPException(status_code=500, detail="Failed to delete upload")


@router.get("/uploads")
async def list_uploads(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all uploads."""
    service = get_file_service()
    files = service.store.list(limit=limit, offset=offset)
    total = service.store.count()

    return {
        "files": [
            {
                "id": f.id,
                "filename": f.filename,
                "size": f.size,
                "mime_type": f.mime_type,
                "media_type": MediaTypeSchema.VIDEO if f.mime_type.startswith("video/") else MediaTypeSchema.IMAGE,
                "created_at": f.created_at,
                "expires_at": f.expires_at,
            }
            for f in files
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/uploads/cleanup")
async def cleanup_uploads():
    """Cleanup expired and orphaned uploads."""
    service = get_file_service()
    expired = service.cleanup_expired()
    orphaned = service.cleanup_orphaned()

    return {
        "expired_deleted": expired,
        "orphaned_deleted": orphaned,
    }


@router.get("/uploads/stats")
async def get_upload_stats():
    """Get upload storage statistics."""
    service = get_file_service()
    return service.get_stats()


@router.get("/uploads/{upload_id}/faces")
async def detect_faces(upload_id: str):
    """Detect faces in an uploaded image.

    Returns face bounding boxes and thumbnail crops for UI display.
    """
    service = get_file_service()
    record = service.get_file(upload_id)

    if not record:
        raise HTTPException(status_code=404, detail="Upload not found")

    path = Path(record.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload file not found")

    if not record.mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Face detection only supported for images")

    # Read image
    image = read_image(path)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to read image")

    # Detect faces
    model_manager = get_model_manager()
    detector = FaceDetector(model_manager=model_manager)
    faces = detector.detect(image)

    # Build response with face thumbnails
    face_data = []
    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Expand bbox slightly for better thumbnail
        h, w = image.shape[:2]
        pad = int((x2 - x1) * 0.2)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Crop face
        face_crop = image[y1:y2, x1:x2]

        # Resize thumbnail to max 128px
        max_dim = 128
        scale = max_dim / max(face_crop.shape[:2])
        if scale < 1:
            new_size = (int(face_crop.shape[1] * scale), int(face_crop.shape[0] * scale))
            face_crop = cv2.resize(face_crop, new_size, interpolation=cv2.INTER_AREA)

        # Encode to base64 JPEG
        _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumb_b64 = base64.b64encode(buffer).decode('utf-8')

        face_data.append({
            "index": idx,
            "bbox": [int(face.bbox[0]), int(face.bbox[1]), int(face.bbox[2]), int(face.bbox[3])],
            "score": float(face.scores.detector),
            "area": int(face.area),
            "thumbnail": f"data:image/jpeg;base64,{thumb_b64}",
        })

    return {
        "upload_id": upload_id,
        "image_size": [image.shape[1], image.shape[0]],
        "face_count": len(faces),
        "faces": face_data,
    }
