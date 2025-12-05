"""Job management routes."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from kaofusion.services.job_service import (
    JobRecord,
    JobStatus,
    get_job_service,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobListResponse(BaseModel):
    """Job list response."""

    jobs: list[dict]
    total: int
    limit: int
    offset: int


class JobStatsResponse(BaseModel):
    """Job statistics response."""

    total: int
    pending: int
    processing: int
    completed: int
    failed: int
    cancelled: int


class JobDetailResponse(BaseModel):
    """Job detail response."""

    id: str
    source_path: str
    target_path: str
    output_path: str | None
    status: str
    progress: float
    error: str | None
    config: dict
    created_at: str
    updated_at: str
    completed_at: str | None


def _record_to_response(record: JobRecord) -> dict:
    """Convert JobRecord to API response dict."""
    return {
        "id": record.id,
        "source_path": record.source_path,
        "target_path": record.target_path,
        "output_path": record.output_path,
        "status": record.status.value,
        "progress": record.progress,
        "error": record.error,
        "config": record.config,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "completed_at": record.completed_at,
        "media_type": record.config.get("media_type", "image"),
    }


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Max jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
):
    """List all jobs with optional filtering.

    Args:
        status: Filter by status (pending, processing, completed, failed, cancelled)
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip for pagination

    Returns:
        List of jobs
    """
    service = get_job_service()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: pending, processing, completed, failed, cancelled",
            )

    jobs = service.list_jobs(status=status_filter, limit=limit, offset=offset)
    total = service.store.count(status_filter)

    return JobListResponse(
        jobs=[_record_to_response(j) for j in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats", response_model=JobStatsResponse)
async def get_job_stats():
    """Get job statistics.

    Returns:
        Job counts by status
    """
    service = get_job_service()
    stats = service.get_stats()
    return JobStatsResponse(**stats)


@router.get("/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str):
    """Get job details.

    Args:
        job_id: Job ID

    Returns:
        Job details
    """
    service = get_job_service()
    record = service.get_job(job_id)

    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    return _record_to_response(record)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or processing job.

    Args:
        job_id: Job ID

    Returns:
        Updated job record
    """
    service = get_job_service()
    record = service.cancel_job(job_id)

    if not record:
        job = service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}",
        )

    return _record_to_response(record)


@router.post("/{job_id}/retry")
async def retry_job(job_id: str):
    """Retry a failed job.

    Args:
        job_id: Job ID

    Returns:
        Updated job record
    """
    service = get_job_service()
    record = service.retry_job(job_id)

    if not record:
        job = service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry job with status: {job.status.value}. Only failed jobs can be retried.",
        )

    return _record_to_response(record)


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its output file.

    Args:
        job_id: Job ID

    Returns:
        Success message
    """
    service = get_job_service()

    if not service.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    if service.delete_job(job_id):
        return {"message": "Job deleted successfully"}

    raise HTTPException(status_code=500, detail="Failed to delete job")


@router.post("/cleanup")
async def cleanup_old_jobs(
    max_age_days: int = Query(7, ge=1, le=365, description="Max age in days"),
):
    """Delete jobs older than max_age_days.

    Args:
        max_age_days: Maximum age in days

    Returns:
        Number of deleted jobs
    """
    service = get_job_service()
    deleted = service.cleanup_old_jobs(max_age_days)
    return {"deleted": deleted}
