"""Face swap routes."""

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from kaofusion.api.routes.uploads import get_upload_path
from kaofusion.api.schemas.job import (
    FaceMaskTypeSchema,
    FaceSelectorModeSchema,
    FaceEnhancerModelSchema,
    FrameEnhancerModelSchema,
    MediaTypeSchema,
    JobCreate,
    JobResponse,
    JobStatus,
    SwapConfigSchema,
)
from kaofusion.config import settings
from kaofusion.processing.types import (
    FaceMaskType,
    FaceSelectorMode,
    FaceEnhancerModel,
    FrameEnhancerModel,
    FaceSwapperModel,
    MediaType,
    OutputFormat,
    SwapConfig,
    VideoConfig,
)
from kaofusion.services.file_service import get_file_service
from kaofusion.services.swap_service import SwapService

router = APIRouter(prefix="/api", tags=["swap"])

# Service instance
_swap_service: SwapService | None = None


def get_swap_service() -> SwapService:
    """Get or create swap service."""
    global _swap_service
    if _swap_service is None:
        _swap_service = SwapService(output_dir=settings.output_dir)
    return _swap_service


def _schema_to_config(schema: SwapConfigSchema | None) -> SwapConfig:
    """Convert API schema to internal SwapConfig.

    Args:
        schema: API config schema or None

    Returns:
        Internal SwapConfig dataclass
    """
    if schema is None:
        return SwapConfig()

    # Map schema enums to internal enums
    mask_types = [FaceMaskType(m.value) for m in schema.mask_types]
    face_selector_mode = FaceSelectorMode(schema.face_selector_mode.value)
    output_format = OutputFormat(schema.output_format.value)
    swapper_model = FaceSwapperModel(schema.swapper_model.value)
    face_enhancer_model = FaceEnhancerModel(schema.face_enhancer_model.value)
    frame_enhancer_model = FrameEnhancerModel(schema.frame_enhancer_model.value)

    video_config = None
    if schema.video:
        video_config = VideoConfig(
            preserve_audio=schema.video.preserve_audio,
            crf=schema.video.crf,
            fps=schema.video.fps,
            encoder=schema.video.encoder,
            max_frames=schema.video.max_frames,
        )

    return SwapConfig(
        detector_score_threshold=schema.detector_score_threshold,
        detector_size=schema.detector_size,
        swapper_model=swapper_model,
        face_selector_mode=face_selector_mode,
        face_selector_index=schema.face_selector_index,
        face_selector_indices=schema.face_selector_indices,
        mask_types=mask_types,
        mask_blur=schema.mask_blur,
        mask_padding=schema.mask_padding,
        output_format=output_format,
        output_quality=schema.output_quality,
        swap_weight=schema.swap_weight,
        face_enhancer_model=face_enhancer_model,
        face_enhancer_blend=schema.face_enhancer_blend,
        face_enhancer_weight=schema.face_enhancer_weight,
        frame_enhancer_model=frame_enhancer_model,
        frame_enhancer_blend=schema.frame_enhancer_blend,
        video=video_config,
        execution_providers=schema.execution_providers,
    )


@router.post("/swap", response_model=JobResponse)
async def create_swap_job(
    request: JobCreate,
    background_tasks: BackgroundTasks,
):
    """Create and start a face swap job.

    Args:
        request: Job creation request
        background_tasks: FastAPI background tasks

    Returns:
        Job info
    """
    file_service = get_file_service()

    source_record = file_service.get_file(request.source_id)
    target_record = file_service.get_file(request.target_id)

    if not source_record:
        raise HTTPException(status_code=404, detail="Source file not found")
    if not target_record:
        raise HTTPException(status_code=404, detail="Target file not found")

    # Get file paths
    source_path = get_upload_path(request.source_id)
    if not source_path or not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")

    target_path = get_upload_path(request.target_id)
    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Target file not found")

    target_is_video = target_record.mime_type.startswith("video/")
    source_is_image = source_record.mime_type.startswith("image/")
    if not source_is_image:
        raise HTTPException(status_code=400, detail="Source must be an image")

    media_type = MediaType.VIDEO if target_is_video else MediaType.IMAGE

    # Convert API schema to internal config
    config = _schema_to_config(request.config)
    if media_type == MediaType.VIDEO and config.video is None:
        config.video = VideoConfig()

    # Create job
    service = get_swap_service()
    job = service.create_job(
        source_path,
        target_path,
        config,
        media_type=media_type,
        output_suffix="mp4" if media_type == MediaType.VIDEO else None,
    )

    # Start processing in background thread
    # execute_job is now sync, perfect for BackgroundTasks
    background_tasks.add_task(service.execute_job, job.id)

    return JobResponse(
        id=job.id,
        status=JobStatus(job.status.value),
        progress=job.progress,
        created_at=job.created_at,
        media_type=MediaTypeSchema(job.media_type.value),
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status (polling endpoint).

    Args:
        job_id: Job ID

    Returns:
        Job info
    """
    service = get_swap_service()
    job = service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    output_url = None
    if job.status.value == "completed" and job.output_path:
        output_url = f"/api/files/{job_id}"

    return JobResponse(
        id=job.id,
        status=JobStatus(job.status.value),
        progress=job.progress,
        media_type=MediaTypeSchema(job.media_type.value),
        frame_count=job.frame_count,
        fps=job.fps,
        output_url=output_url,
        error=job.error,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get("/jobs/{job_id}/stream")
async def stream_job_status(job_id: str):
    """Stream job status via Server-Sent Events.

    This endpoint keeps the connection open and sends progress updates
    in real-time until the job completes or fails.

    Args:
        job_id: Job ID

    Returns:
        SSE stream with progress events
    """
    service = get_swap_service()

    async def event_generator():
        """Generate SSE events for job progress."""
        last_progress = -1.0

        while True:
            job = service.get_job(job_id)

            if not job:
                yield _sse_event("error", {"error": "Job not found"})
                break

            # Only send update if progress changed
            if job.progress != last_progress:
                last_progress = job.progress
                yield _sse_event("progress", {
                    "progress": job.progress,
                    "status": job.status.value,
                    "media_type": job.media_type.value,
                })

            if job.status.value == "completed":
                yield _sse_event("complete", {
                    "output_url": f"/api/files/{job_id}",
                    "progress": 1.0,
                    "media_type": job.media_type.value,
                })
                break
            elif job.status.value == "failed":
                yield _sse_event("error", {
                    "error": job.error or "Unknown error",
                    "media_type": job.media_type.value,
                })
                break

            # Check every 200ms for responsive updates
            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


def _sse_event(event_type: str, data: dict) -> str:
    """Format data as SSE event.

    Args:
        event_type: Event name (progress, complete, error)
        data: Event data dict

    Returns:
        Formatted SSE string
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.get("/files/{job_id}")
async def download_result(job_id: str):
    """Download job result.

    Args:
        job_id: Job ID

    Returns:
        Result image file
    """
    service = get_swap_service()
    job = service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status.value != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    if not job.output_path or not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    # Determine media type based on output format
    ext = job.output_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
    }
    media_type = media_types.get(ext, "image/png")

    return FileResponse(
        job.output_path,
        media_type=media_type,
        filename=f"swap_{job_id}{ext}",
    )
