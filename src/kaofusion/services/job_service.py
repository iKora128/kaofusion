"""Job management service with persistence.

Provides job tracking, history, and persistence across server restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobRecord:
    """Persistent job record."""

    id: str
    source_path: str
    target_path: str
    output_path: str | None = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobStore:
    """File-based job storage."""

    def __init__(self, storage_dir: Path):
        """Initialize job store.

        Args:
            storage_dir: Directory for job metadata files
        """
        self.storage_dir = storage_dir
        self.jobs_file = storage_dir / "jobs.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load jobs from disk."""
        if self.jobs_file.exists():
            try:
                with open(self.jobs_file, "r") as f:
                    data = json.load(f)
                self._jobs = {
                    k: JobRecord.from_dict(v)
                    for k, v in data.items()
                }
                logger.info(f"Loaded {len(self._jobs)} jobs from storage")
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")
                self._jobs = {}
        else:
            self._jobs = {}

    def _save(self) -> None:
        """Save jobs to disk."""
        try:
            data = {k: v.to_dict() for k, v in self._jobs.items()}
            with open(self.jobs_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def get(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of job records
        """
        jobs = list(self._jobs.values())

        # Filter by status
        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply pagination
        return jobs[offset : offset + limit]

    def create(self, record: JobRecord) -> JobRecord:
        """Create a new job record."""
        self._jobs[record.id] = record
        self._save()
        return record

    def update(self, job_id: str, **updates: Any) -> JobRecord | None:
        """Update a job record.

        Args:
            job_id: Job ID
            **updates: Fields to update

        Returns:
            Updated record or None if not found
        """
        record = self._jobs.get(job_id)
        if not record:
            return None

        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)

        record.updated_at = datetime.now().isoformat()
        self._save()
        return record

    def delete(self, job_id: str) -> bool:
        """Delete a job record."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def count(self, status: JobStatus | None = None) -> int:
        """Count jobs with optional status filter."""
        if status:
            return sum(1 for j in self._jobs.values() if j.status == status)
        return len(self._jobs)


class JobService:
    """High-level job management service."""

    def __init__(self, storage_dir: Path | None = None):
        """Initialize job service.

        Args:
            storage_dir: Directory for job storage
        """
        self.storage_dir = storage_dir or Path.cwd() / ".kaofusion" / "jobs"
        self.store = JobStore(self.storage_dir)

    def create_job(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        config: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Create a new job.

        Args:
            source_path: Path to source image
            target_path: Path to target image
            output_path: Path for output image
            config: Optional configuration dict

        Returns:
            Created job record
        """
        job_id = str(uuid.uuid4())[:8]
        record = JobRecord(
            id=job_id,
            source_path=str(source_path),
            target_path=str(target_path),
            output_path=str(output_path),
            config=config or {},
        )
        return self.store.create(record)

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get a job by ID."""
        return self.store.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[JobRecord]:
        """List jobs with optional filtering."""
        return self.store.list(status=status, limit=limit, offset=offset)

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress: float | None = None,
        error: str | None = None,
    ) -> JobRecord | None:
        """Update job status.

        Args:
            job_id: Job ID
            status: New status
            progress: Optional progress value
            error: Optional error message

        Returns:
            Updated record or None if not found
        """
        updates: dict[str, Any] = {"status": status}
        if progress is not None:
            updates["progress"] = progress
        if error is not None:
            updates["error"] = error
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            updates["completed_at"] = datetime.now().isoformat()
        return self.store.update(job_id, **updates)

    def update_job_progress(self, job_id: str, progress: float) -> JobRecord | None:
        """Update job progress."""
        return self.store.update(job_id, progress=progress)

    def cancel_job(self, job_id: str) -> JobRecord | None:
        """Cancel a job.

        Only pending or processing jobs can be cancelled.
        """
        record = self.store.get(job_id)
        if not record:
            return None

        if record.status not in (JobStatus.PENDING, JobStatus.PROCESSING):
            logger.warning(f"Cannot cancel job {job_id} with status {record.status}")
            return None

        return self.update_job_status(job_id, JobStatus.CANCELLED)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job and its output file."""
        record = self.store.get(job_id)
        if not record:
            return False

        # Delete output file if exists
        if record.output_path:
            output_path = Path(record.output_path)
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete output file: {e}")

        return self.store.delete(job_id)

    def retry_job(self, job_id: str) -> JobRecord | None:
        """Reset a failed job to pending status for retry.

        Only failed jobs can be retried.
        """
        record = self.store.get(job_id)
        if not record:
            return None

        if record.status != JobStatus.FAILED:
            logger.warning(f"Cannot retry job {job_id} with status {record.status}")
            return None

        return self.store.update(
            job_id,
            status=JobStatus.PENDING,
            progress=0.0,
            error=None,
            completed_at=None,
        )

    def cleanup_old_jobs(self, max_age_days: int = 7) -> int:
        """Delete jobs older than max_age_days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of deleted jobs
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted = 0

        for job in self.store.list(limit=10000):
            created = datetime.fromisoformat(job.created_at)
            if created < cutoff:
                if self.delete_job(job.id):
                    deleted += 1

        logger.info(f"Cleaned up {deleted} old jobs")
        return deleted

    def get_stats(self) -> dict[str, int]:
        """Get job statistics."""
        return {
            "total": self.store.count(),
            "pending": self.store.count(JobStatus.PENDING),
            "processing": self.store.count(JobStatus.PROCESSING),
            "completed": self.store.count(JobStatus.COMPLETED),
            "failed": self.store.count(JobStatus.FAILED),
            "cancelled": self.store.count(JobStatus.CANCELLED),
        }


# Global instance
_job_service: JobService | None = None


def get_job_service() -> JobService:
    """Get the global job service instance."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service
