"""Benchmark routes for performance testing."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import APIRouter, HTTPException

from kaofusion.config import settings
from kaofusion.processing.models import (
    MODELS,
    ModelManager,
    detect_execution_providers,
    get_model_manager,
)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


@router.get("/system")
async def get_system_info():
    """Get system information for benchmarking.

    Returns:
        System info including CPU, memory, and available providers
    """
    import platform
    import psutil

    cpu_info = {
        "count_physical": psutil.cpu_count(logical=False),
        "count_logical": psutil.cpu_count(logical=True),
        "percent": psutil.cpu_percent(interval=0.1),
    }

    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "percent_used": memory.percent,
    }

    available_providers = ort.get_available_providers()
    detected_providers = detect_execution_providers()

    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "cpu": cpu_info,
        "memory": memory_info,
        "onnx_available_providers": available_providers,
        "onnx_detected_providers": detected_providers,
    }


@router.get("/models")
async def list_available_models():
    """List all available models with their specifications.

    Returns:
        List of model specs
    """
    return {
        "models": [
            {
                "name": spec.name,
                "filename": spec.filename,
                "input_size": spec.input_size,
                "release_tag": spec.release_tag,
            }
            for spec in MODELS.values()
        ]
    }


@router.get("/models/{model_name}")
async def benchmark_model(model_name: str, iterations: int = 10):
    """Benchmark a specific model.

    Args:
        model_name: Name of the model to benchmark
        iterations: Number of iterations to run

    Returns:
        Benchmark results including avg/min/max times
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    spec = MODELS[model_name]
    manager = get_model_manager()

    # Ensure model is loaded
    try:
        session = manager.get_session(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Get input shape
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape

    # Create random input (handle dynamic dimensions)
    shape = []
    for dim in input_shape:
        if isinstance(dim, str) or dim is None:
            # Dynamic dimension, use spec size or default
            if spec.input_size:
                shape.append(spec.input_size[0] if len(shape) < 2 else spec.input_size[1])
            else:
                shape.append(256)
        else:
            shape.append(dim)

    # Create input tensor
    dtype = np.float32
    dummy_input = np.random.rand(*shape).astype(dtype)

    # Warm up
    for _ in range(3):
        session.run(None, {input_info.name: dummy_input})

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_info.name: dummy_input})
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        "model": model_name,
        "iterations": iterations,
        "input_shape": shape,
        "execution_provider": session.get_providers()[0],
        "times_ms": {
            "avg": round(np.mean(times), 2),
            "min": round(np.min(times), 2),
            "max": round(np.max(times), 2),
            "std": round(np.std(times), 2),
        },
        "throughput_fps": round(1000 / np.mean(times), 2),
    }


@router.get("/pipeline")
async def benchmark_pipeline(iterations: int = 3):
    """Benchmark the full face swap pipeline.

    Args:
        iterations: Number of iterations per model

    Returns:
        Pipeline benchmark results
    """
    manager = get_model_manager()
    results = {}

    pipeline_models = ["retinaface_10g", "2dfan4", "arcface_w600k_r50", "inswapper_128"]

    for model_name in pipeline_models:
        if model_name not in MODELS:
            continue

        try:
            spec = MODELS[model_name]
            session = manager.get_session(model_name)
            input_info = session.get_inputs()[0]
            input_shape = input_info.shape

            # Create input
            shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim is None:
                    if spec.input_size:
                        shape.append(
                            spec.input_size[0] if len(shape) < 2 else spec.input_size[1]
                        )
                    else:
                        shape.append(256)
                else:
                    shape.append(dim)

            dummy_input = np.random.rand(*shape).astype(np.float32)

            # Warm up
            session.run(None, {input_info.name: dummy_input})

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                session.run(None, {input_info.name: dummy_input})
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)

            results[model_name] = {
                "avg_ms": round(np.mean(times), 2),
                "min_ms": round(np.min(times), 2),
                "max_ms": round(np.max(times), 2),
            }

        except Exception as e:
            results[model_name] = {"error": str(e)}

    # Calculate total
    total_avg = sum(r.get("avg_ms", 0) for r in results.values() if "avg_ms" in r)

    return {
        "iterations": iterations,
        "models": results,
        "total_avg_ms": round(total_avg, 2),
        "estimated_fps": round(1000 / total_avg, 2) if total_avg > 0 else 0,
    }
