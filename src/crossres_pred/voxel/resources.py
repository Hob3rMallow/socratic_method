from __future__ import annotations

import math
import os
import subprocess

import torch

MAX_PROJECT_CPU_THREADS = 16
MAX_GPU_POWER_LIMIT_WATTS = 600.5


def configure_cpu_budget(max_cpu_threads: int, *, reserve_processes: int = 0) -> int:
    """Cap numerical CPU work while leaving room for loader processes.

    The Windows launch wrapper additionally pins the complete process tree to
    logical CPUs 0-15. This in-process guard prevents Torch and BLAS pools from
    independently expanding to the host's full 128 logical CPUs.
    """

    if not 1 <= max_cpu_threads <= MAX_PROJECT_CPU_THREADS:
        raise ValueError(f"max_cpu_threads must be in [1, {MAX_PROJECT_CPU_THREADS}]")
    if reserve_processes < 0 or reserve_processes >= max_cpu_threads:
        raise ValueError("reserve_processes must be in [0, max_cpu_threads)")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch_threads = max(1, max_cpu_threads - reserve_processes)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return torch_threads


def assert_cuda_power_limit(
    device: torch.device,
    *,
    maximum_watts: float = MAX_GPU_POWER_LIMIT_WATTS,
) -> float | None:
    if device.type != "cuda":
        return None
    device_index = device.index if device.index is not None else 0
    command = [
        "nvidia-smi",
        "-i",
        str(device_index),
        "--query-gpu=power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt"
                else 0
            ),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("could not query the CUDA GPU power limit") from error
    if result.returncode != 0 or not result.stdout.strip():
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"could not query the CUDA GPU power limit: {detail}")
    try:
        limit = float(result.stdout.splitlines()[0].strip())
    except ValueError as error:
        raise RuntimeError(
            f"invalid CUDA GPU power limit: {result.stdout!r}"
        ) from error
    if not math.isfinite(limit) or limit > maximum_watts:
        raise RuntimeError(
            f"CUDA GPU {device_index} power limit is {limit} W; "
            f"required maximum is {maximum_watts} W"
        )
    return limit
