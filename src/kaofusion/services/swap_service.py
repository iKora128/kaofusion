"""Face swap service.

Orchestrates the face swap pipeline.
"""

import copy
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kaofusion.processing.detector import FaceDetector
from kaofusion.processing.landmarker import FaceLandmarker, detect_landmarks
from kaofusion.processing.models import ModelManager, get_model_manager
from kaofusion.processing.postprocess import (
    FaceEnhancer,
    FrameEnhancer,
    PostProcessorChain,
)
from kaofusion.processing.recognizer import FaceRecognizer
from kaofusion.processing.swapper import FaceSwapper
from kaofusion.processing.types import (
    Face,
    FaceSelectorMode,
    MediaType,
    OutputFormat,
    SwapConfig,
    VideoConfig,
)
from kaofusion.processing.vision import read_image, write_image

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SwapJob:
    """Face swap job."""

    id: str
    source_path: Path
    target_path: Path
    config: SwapConfig = field(default_factory=SwapConfig)
    output_path: Path | None = None
    media_type: MediaType = MediaType.IMAGE
    frame_count: int | None = None
    fps: float | None = None
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    details: dict[str, Any] = field(default_factory=dict)


class SwapService:
    """Service for executing face swap operations."""

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        output_dir: Path | None = None,
    ):
        """Initialize swap service.

        Args:
            model_manager: Model manager instance
            output_dir: Directory for output files
        """
        self.model_manager = model_manager or get_model_manager()
        self.output_dir = output_dir or Path.cwd() / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.detector = FaceDetector(model_manager=self.model_manager)
        self.landmarker = FaceLandmarker(model_manager=self.model_manager)
        self.recognizer = FaceRecognizer(model_manager=self.model_manager)
        self.swapper = FaceSwapper(model_manager=self.model_manager)
        self.postprocessors = PostProcessorChain(
            face_enhancer=FaceEnhancer(self.model_manager),
            frame_enhancer=FrameEnhancer(self.model_manager),
        )

        # Job tracking
        self._jobs: dict[str, SwapJob] = {}

    def create_job(
        self,
        source_path: Path,
        target_path: Path,
        config: SwapConfig | None = None,
        media_type: MediaType = MediaType.IMAGE,
        output_suffix: str | None = None,
    ) -> SwapJob:
        """Create a new swap job.

        Args:
            source_path: Path to source face image
            target_path: Path to target image
            config: Optional swap configuration
            media_type: Media type (image or video)
            output_suffix: Optional override for output extension

        Returns:
            Created job
        """
        job_id = str(uuid.uuid4())[:8]
        config = config or SwapConfig()

        # Determine output extension based on format
        if media_type == MediaType.VIDEO:
            ext = output_suffix or "mp4"
        else:
            ext = output_suffix or config.output_format.value
        output_path = self.output_dir / f"swap_{job_id}.{ext}"

        job = SwapJob(
            id=job_id,
            source_path=source_path,
            target_path=target_path,
            config=config,
            output_path=output_path,
            media_type=media_type,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> SwapJob | None:
        """Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job or None if not found
        """
        return self._jobs.get(job_id)

    def execute_job(self, job_id: str) -> SwapJob:
        """Execute a swap job synchronously.

        This is designed to be called from BackgroundTasks or a thread pool.
        For CPU-bound ONNX inference, sync execution in a background thread
        is cleaner than async wrappers.

        Args:
            job_id: Job ID

        Returns:
            Updated job

        Raises:
            ValueError: If job not found
        """
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.status = JobStatus.PROCESSING
        job.progress = 0.0

        try:
            if job.media_type == MediaType.VIDEO:
                result_path, frame_count, fps = self._execute_video_swap(
                    job.source_path,
                    job.target_path,
                    job.output_path,
                    job.config,
                    lambda p: self._update_progress(job_id, p),
                )
                job.frame_count = frame_count
                job.fps = fps
                if frame_count is not None:
                    job.details["frame_count"] = frame_count
                if fps is not None:
                    job.details["fps"] = fps
            else:
                result_path = self._execute_swap(
                    job.source_path,
                    job.target_path,
                    job.output_path,
                    job.config,
                    lambda p: self._update_progress(job_id, p),
                )

            job.status = JobStatus.COMPLETED
            job.output_path = result_path
            job.progress = 1.0
            job.completed_at = datetime.now()

        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            job.status = JobStatus.FAILED
            job.error = str(e)

        return job

    def _update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress.

        Args:
            job_id: Job ID
            progress: Progress value (0-1)
        """
        job = self._jobs.get(job_id)
        if job:
            job.progress = progress

    def _select_faces(
        self,
        faces: list[Face],
        source_face: Face,
        config: SwapConfig,
    ) -> list[Face]:
        """Select which faces to swap based on configuration.

        Args:
            faces: All detected faces
            source_face: Source face for similarity comparison
            config: Swap configuration

        Returns:
            List of faces to swap
        """
        if not faces:
            return []

        mode = config.face_selector_mode

        if mode == FaceSelectorMode.ALL:
            return faces

        if mode == FaceSelectorMode.LARGEST:
            # Return face with largest area
            return [max(faces, key=lambda f: f.area)]

        if mode == FaceSelectorMode.BY_INDEX:
            # Return specific face by index
            index = config.face_selector_index or 0
            if 0 <= index < len(faces):
                return [faces[index]]
            logger.warning(f"Face index {index} out of range, using first face")
            return [faces[0]]

        if mode == FaceSelectorMode.BY_INDICES:
            # Return multiple faces by indices
            indices = config.face_selector_indices or [0]
            selected = []
            for idx in indices:
                if 0 <= idx < len(faces):
                    selected.append(faces[idx])
                else:
                    logger.warning(f"Face index {idx} out of range, skipping")
            if not selected:
                logger.warning("No valid face indices, using first face")
                return [faces[0]]
            return selected

        if mode == FaceSelectorMode.BEST_MATCH:
            # Return face most similar to source (by embedding cosine similarity)
            if source_face.embedding_norm is None:
                logger.warning("Source face has no embedding, using first face")
                return [faces[0]]

            best_face = None
            best_similarity = -1.0

            for face in faces:
                if face.embedding_norm is not None:
                    # Cosine similarity
                    similarity = float(
                        np.dot(source_face.embedding_norm, face.embedding_norm)
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_face = face

            if best_face is None:
                logger.warning("No face has embedding, using first face")
                return [faces[0]]

            return [best_face]

        return faces

    def _apply_detector_settings(self, config: SwapConfig) -> None:
        """Apply detector settings from config."""
        self.detector.score_threshold = config.detector_score_threshold
        self.detector.input_size = config.detector_size

    def _prepare_source_face(
        self,
        source_image: np.ndarray,
        config: SwapConfig,
    ) -> Face:
        """Detect and embed the source face once (reused for video frames)."""
        self._apply_detector_settings(config)
        source_faces = self.detector.detect(source_image)
        if not source_faces:
            raise ValueError("No face detected in source image")
        source_face = source_faces[0]

        source_landmarks = detect_landmarks(
            source_image,
            source_face.bbox,
            self.model_manager,
        )
        if source_landmarks:
            source_face.landmarks = source_landmarks

        embedding, embedding_norm = self.recognizer.compute_embedding(
            source_image,
            source_face.landmarks.five,
        )
        source_face.embedding = embedding
        source_face.embedding_norm = embedding_norm
        return source_face

    def _process_target_faces(
        self,
        target_image: np.ndarray,
        source_face: Face,
        config: SwapConfig,
        progress_callback: Any = None,
        progress_start: float = 0.0,
        progress_span: float = 1.0,
        cached_faces: list[Face] | None = None,
        skip_detection: bool = False,
    ) -> tuple[np.ndarray, list[Face]]:
        """Run detection, selection, swap, and return swapped image + faces."""
        if skip_detection and cached_faces:
            # Reuse cached detections (optional, for video speed-up)
            target_faces = [copy.deepcopy(f) for f in cached_faces]
        else:
            target_faces = self.detector.detect(target_image)
            if not target_faces:
                raise ValueError("No face detected in target image")

        for target_face in target_faces:
            target_landmarks = detect_landmarks(
                target_image,
                target_face.bbox,
                self.model_manager,
            )
            if target_landmarks:
                target_face.landmarks = target_landmarks

            t_embedding, t_embedding_norm = self.recognizer.compute_embedding(
                target_image,
                target_face.landmarks.five,
            )
            target_face.embedding = t_embedding
            target_face.embedding_norm = t_embedding_norm

        selected_faces = self._select_faces(target_faces, source_face, config)
        if not selected_faces:
            raise ValueError("No faces selected for swapping")

        logger.info(
            f"Selected {len(selected_faces)} of {len(target_faces)} faces for swapping"
        )

        self.swapper.mask_blur = config.mask_blur
        self.swapper.mask_padding = config.mask_padding

        result_image = target_image.copy()
        swapped_faces: list[Face] = []

        for i, target_face in enumerate(selected_faces):
            result_image = self.swapper.swap(
                source_face,
                target_face,
                result_image,
                config,
            )
            swapped_faces.append(target_face)

            if progress_callback:
                progress = progress_start + progress_span * (i + 1) / max(
                    len(selected_faces), 1
                )
                progress_callback(progress)

        return result_image, swapped_faces

    def _probe_video(self, video_path: Path) -> dict[str, Any]:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning("ffprobe failed: %s", result.stderr.decode() if result.stderr else "unknown")
            return {}
        try:
            data = json.loads(result.stdout.decode())
            stream = (data.get("streams") or [{}])[0]
            fps = None
            if "r_frame_rate" in stream and stream["r_frame_rate"]:
                num, den = stream["r_frame_rate"].split("/")
                fps = float(num) / float(den) if float(den) else None
            nb_frames = None
            if stream.get("nb_frames"):
                try:
                    nb_frames = int(stream["nb_frames"])
                except ValueError:
                    nb_frames = None
            return {
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": fps,
                "nb_frames": nb_frames,
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse ffprobe output: %s", exc)
            return {}

    def _compute_process_size(self, width: int, height: int, video_cfg: VideoConfig) -> tuple[int, int]:
        """Compute processing resolution with optional downscale."""
        max_side = video_cfg.max_side
        if not max_side:
            return width, height
        scale = min(1.0, max_side / max(width, height))
        if scale >= 1.0:
            return width, height
        new_w = int(width * scale)
        new_h = int(height * scale)
        return max(1, new_w), max(1, new_h)

    def _compute_progress(
        self,
        processed: int,
        total_frames: int | None,
        start_time: float,
        fps_hint: float = 24.0,
    ) -> float:
        """Estimate progress with optional fallback when frame count unknown."""
        if total_frames and total_frames > 0:
            return min(0.98, 0.05 + 0.9 * processed / max(total_frames, 1))
        elapsed = max(time.time() - start_time, 0.001)
        estimated_total = max(processed, int(elapsed * fps_hint))
        if estimated_total == 0:
            return 0.05
        return min(0.98, 0.05 + 0.9 * processed / estimated_total)

    def _execute_swap(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        config: SwapConfig,
        progress_callback: Any = None,
    ) -> Path:
        """Execute face swap synchronously.

        Args:
            source_path: Path to source face image
            target_path: Path to target image
            output_path: Path for output image
            config: Swap configuration
            progress_callback: Optional progress callback

        Returns:
            Path to output image
        """
        if progress_callback:
            progress_callback(0.05)

        source_image = read_image(source_path)
        if source_image is None:
            raise ValueError(f"Failed to read source image: {source_path}")

        target_image = read_image(target_path)
        if target_image is None:
            raise ValueError(f"Failed to read target image: {target_path}")

        if progress_callback:
            progress_callback(0.1)

        source_face = self._prepare_source_face(source_image, config)

        if progress_callback:
            progress_callback(0.3)

        result_image, swapped_faces = self._process_target_faces(
            target_image,
            source_face,
            config,
            progress_callback=progress_callback,
            progress_start=0.35,
            progress_span=0.45,
        )

        # Post-process (face enhancer -> frame enhancer)
        result_image = self.postprocessors.apply(result_image, swapped_faces, config)

        # Save result with quality settings
        quality = (
            config.output_quality
            if config.output_format != OutputFormat.PNG
            else None
        )
        if not write_image(output_path, result_image, quality=quality):
            raise ValueError(f"Failed to write output image: {output_path}")

        if progress_callback:
            progress_callback(1.0)

        return output_path

    def _execute_video_swap(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path,
        config: SwapConfig,
        progress_callback: Any = None,
    ) -> tuple[Path, int | None, float | None]:
        """Process video using ffmpeg streaming (single encode, optional redetect skip)."""
        if progress_callback:
            progress_callback(0.02)

        if config.video is None:
            config.video = VideoConfig()
        video_cfg = config.video

        source_image = read_image(source_path)
        if source_image is None:
            raise ValueError(f"Failed to read source image: {source_path}")
        source_face = self._prepare_source_face(source_image, config)

        probe = self._probe_video(target_path)
        width = probe.get("width")
        height = probe.get("height")
        total_frames = probe.get("nb_frames")
        fps = video_cfg.fps or probe.get("fps") or 24.0
        if not width or not height:
            raise ValueError("Could not determine video dimensions for processing")

        # Optional downscale for processing only (output keeps original size)
        proc_width, proc_height = self._compute_process_size(width, height, video_cfg)
        scale_back = (proc_width != width) or (proc_height != height)

        decode_cmd = [
            "ffmpeg",
            "-v",
            "error",
        ]
        if video_cfg.hwaccel:
            decode_cmd += ["-hwaccel", video_cfg.hwaccel]
        if video_cfg.decoder:
            decode_cmd += ["-c:v", video_cfg.decoder]
        decode_cmd += [
            "-i",
            str(target_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]

        encode_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-i",
            str(target_path),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            video_cfg.encoder,
            "-crf",
            str(video_cfg.crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
        ]
        if not video_cfg.preserve_audio:
            encode_cmd += ["-an"]
        else:
            encode_cmd += ["-c:a", "copy"]
        encode_cmd += [
            "-shortest",
            str(output_path),
        ]

        decode_proc = subprocess.Popen(
            decode_cmd,
            stdout=subprocess.PIPE,
        )
        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        frame_size = width * height * 3
        processed = 0
        start_time = time.time()
        last_faces: list[Face] | None = None
        redetect_interval = max(video_cfg.redetect_interval, 0)

        try:
            while True:
                raw_frame = decode_proc.stdout.read(frame_size) if decode_proc.stdout else b""
                if not raw_frame or len(raw_frame) < frame_size:
                    break

                if video_cfg.max_frames and processed >= video_cfg.max_frames:
                    break

                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                work_frame = frame
                if scale_back:
                    work_frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)

                try:
                    use_cached = (
                        redetect_interval > 0
                        and last_faces
                        and (processed % redetect_interval != 0)
                    )
                    frame_result, swapped_faces = self._process_target_faces(
                        work_frame,
                        source_face,
                        config,
                        progress_callback=None,
                        cached_faces=last_faces if use_cached else None,
                        skip_detection=use_cached,
                    )
                    frame_result = self.postprocessors.apply(
                        frame_result, swapped_faces, config
                    )
                    last_faces = swapped_faces
                except Exception as frame_error:
                    logger.warning(f"Skipping frame {processed} due to error: {frame_error}")
                    frame_result = work_frame

                if scale_back and frame_result.shape[:2] != (height, width):
                    frame_result = cv2.resize(frame_result, (width, height), interpolation=cv2.INTER_CUBIC)

                if encode_proc.stdin:
                    encode_proc.stdin.write(frame_result.astype(np.uint8).tobytes())

                processed += 1
                if progress_callback:
                    progress = self._compute_progress(
                        processed=processed,
                        total_frames=total_frames,
                        start_time=start_time,
                        fps_hint=video_cfg.progress_fps_hint,
                    )
                    progress_callback(progress)
        finally:
            if decode_proc.stdout:
                decode_proc.stdout.close()
            if encode_proc.stdin:
                encode_proc.stdin.close()

        decode_return = decode_proc.wait()
        if decode_return != 0:
            raise RuntimeError(f"ffmpeg decode failed with code {decode_return}")

        # Wait for encode process to finish (stdin already closed)
        # Read stderr before waiting to avoid potential deadlock
        enc_stderr = encode_proc.stderr.read() if encode_proc.stderr else b""
        if encode_proc.stderr:
            encode_proc.stderr.close()
        if encode_proc.stdout:
            encode_proc.stdout.close()

        encode_return = encode_proc.wait()
        if encode_return != 0:
            err_msg = enc_stderr.decode() if enc_stderr else "unknown"
            raise RuntimeError(f"ffmpeg encode failed: {err_msg}")

        if progress_callback:
            progress_callback(1.0)

        return output_path, processed if processed > 0 else total_frames, fps

    def _mux_audio(
        self,
        source_video: Path,
        video_only: Path,
        output_path: Path,
        video_config: VideoConfig,
    ) -> None:
        """Mux original audio track with processed video using ffmpeg."""
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_only),
            "-i",
            str(source_video),
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c:v",
            video_config.encoder,
            "-crf",
            str(video_config.crf),
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.error(
                "ffmpeg muxing failed with code %s: %s",
                result.returncode,
                result.stderr.decode() if result.stderr else "unknown",
            )
            raise RuntimeError("ffmpeg muxing failed")

    def swap_faces(
        self,
        source_path: Path,
        target_path: Path,
        output_path: Path | None = None,
        config: SwapConfig | None = None,
    ) -> Path:
        """Synchronous face swap convenience method.

        Args:
            source_path: Path to source face image
            target_path: Path to target image
            output_path: Optional output path
            config: Optional swap configuration

        Returns:
            Path to output image
        """
        config = config or SwapConfig()
        if output_path is None:
            ext = config.output_format.value
            output_path = self.output_dir / f"swap_{uuid.uuid4().hex[:8]}.{ext}"

        return self._execute_swap(source_path, target_path, output_path, config)
