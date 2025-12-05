"""File management service with persistence.

Provides file upload tracking, persistence, and cleanup.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FileRecord:
    """Persistent file record."""

    id: str
    filename: str
    path: str
    size: int
    mime_type: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str | None = None  # ISO format datetime when file should be deleted

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileRecord":
        """Create from dictionary."""
        return cls(**data)

    @property
    def is_expired(self) -> bool:
        """Check if file has expired."""
        if not self.expires_at:
            return False
        return datetime.now() > datetime.fromisoformat(self.expires_at)


class FileStore:
    """File-based file metadata storage."""

    def __init__(self, storage_dir: Path):
        """Initialize file store.

        Args:
            storage_dir: Directory for metadata file
        """
        self.storage_dir = storage_dir
        self.files_file = storage_dir / "files.json"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, FileRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load files from disk."""
        if self.files_file.exists():
            try:
                with open(self.files_file, "r") as f:
                    data = json.load(f)
                self._files = {k: FileRecord.from_dict(v) for k, v in data.items()}
                logger.info(f"Loaded {len(self._files)} file records from storage")
            except Exception as e:
                logger.error(f"Failed to load files: {e}")
                self._files = {}
        else:
            self._files = {}

    def _save(self) -> None:
        """Save files to disk."""
        try:
            data = {k: v.to_dict() for k, v in self._files.items()}
            with open(self.files_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save files: {e}")

    def get(self, file_id: str) -> FileRecord | None:
        """Get a file by ID."""
        return self._files.get(file_id)

    def list(self, limit: int = 100, offset: int = 0) -> list[FileRecord]:
        """List files with pagination."""
        files = list(self._files.values())
        files.sort(key=lambda f: f.created_at, reverse=True)
        return files[offset : offset + limit]

    def create(self, record: FileRecord) -> FileRecord:
        """Create a new file record."""
        self._files[record.id] = record
        self._save()
        return record

    def delete(self, file_id: str) -> bool:
        """Delete a file record."""
        if file_id in self._files:
            del self._files[file_id]
            self._save()
            return True
        return False

    def count(self) -> int:
        """Count total files."""
        return len(self._files)

    def get_expired(self) -> list[FileRecord]:
        """Get all expired file records."""
        return [f for f in self._files.values() if f.is_expired]


class FileService:
    """High-level file management service."""

    def __init__(
        self,
        upload_dir: Path | None = None,
        storage_dir: Path | None = None,
        default_ttl_hours: int = 24,
    ):
        """Initialize file service.

        Args:
            upload_dir: Directory for uploaded files
            storage_dir: Directory for metadata storage
            default_ttl_hours: Default time-to-live for uploads in hours
        """
        self.upload_dir = upload_dir or Path.cwd() / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.storage_dir = storage_dir or Path.cwd() / ".kaofusion" / "files"
        self.store = FileStore(self.storage_dir)
        self.default_ttl_hours = default_ttl_hours

    def register_file(
        self,
        file_id: str,
        filename: str,
        path: Path,
        size: int,
        mime_type: str,
        ttl_hours: int | None = None,
    ) -> FileRecord:
        """Register a new file.

        Args:
            file_id: Unique file ID
            filename: Original filename
            path: File path
            size: File size in bytes
            mime_type: MIME type
            ttl_hours: Time-to-live in hours (None for default, 0 for no expiry)

        Returns:
            Created file record
        """
        # Calculate expiry time
        expires_at = None
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
        if ttl > 0:
            expires_at = (datetime.now() + timedelta(hours=ttl)).isoformat()

        record = FileRecord(
            id=file_id,
            filename=filename,
            path=str(path),
            size=size,
            mime_type=mime_type,
            expires_at=expires_at,
        )
        return self.store.create(record)

    def get_file(self, file_id: str) -> FileRecord | None:
        """Get a file by ID."""
        return self.store.get(file_id)

    def get_file_path(self, file_id: str) -> Path | None:
        """Get file path by ID."""
        record = self.store.get(file_id)
        if record:
            path = Path(record.path)
            if path.exists():
                return path
        return None

    def delete_file(self, file_id: str) -> bool:
        """Delete a file and its record.

        Args:
            file_id: File ID

        Returns:
            True if deleted successfully
        """
        record = self.store.get(file_id)
        if not record:
            return False

        # Delete actual file
        path = Path(record.path)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete file {path}: {e}")

        # Delete record
        return self.store.delete(file_id)

    def cleanup_expired(self) -> int:
        """Delete all expired files.

        Returns:
            Number of deleted files
        """
        expired = self.store.get_expired()
        deleted = 0

        for record in expired:
            if self.delete_file(record.id):
                deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired files")

        return deleted

    def cleanup_orphaned(self) -> int:
        """Delete files on disk that have no record.

        Returns:
            Number of deleted orphaned files
        """
        deleted = 0
        known_paths = {Path(r.path) for r in self.store.list(limit=10000)}

        for file_path in self.upload_dir.iterdir():
            if file_path.is_file() and file_path not in known_paths:
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete orphaned file {file_path}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} orphaned files")

        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get file storage statistics."""
        files = self.store.list(limit=10000)
        total_size = sum(f.size for f in files)
        expired_count = len(self.store.get_expired())

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "expired_count": expired_count,
        }


# Global instance
_file_service: FileService | None = None


def get_file_service() -> FileService:
    """Get the global file service instance."""
    global _file_service
    if _file_service is None:
        from kaofusion.config import settings

        _file_service = FileService(
            upload_dir=settings.upload_dir,
        )
    return _file_service
