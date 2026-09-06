"""Explicit batch-size semantics shared by all pipeline stages."""

import os

import torch.distributed as dist


def launched_world_size(fallback=1):
    """Prefer the actual process group, then torchrun/SLURM metadata."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        if key in os.environ:
            size = int(os.environ[key])
            if size < 1:
                raise ValueError(f"{key} must be positive")
            return size
    if fallback < 1:
        raise ValueError("world size must be positive")
    return fallback


def resolve_batch_size(batch_size, world_size, mode=None):
    """Return auditable per-device and global batch sizes, without accumulation.

    Global mode preserves the requested batch, not a rounded approximation.
    Per-device mode preserves local GPU work while the global batch grows.
    """
    if batch_size < 1 or world_size < 1:
        raise ValueError("batch_size and world_size must be positive")
    if mode is None:
        evaluation = os.getenv("PERFORM_EVALUATION", "false").lower() == "true"
        mode = (
            os.getenv("EVALUATION_BATCH_SIZE_MODE", "per_device")
            if evaluation
            else os.getenv("BATCH_SIZE_MODE", "global")
        )
    if mode not in ("global", "per_device"):
        raise ValueError("batch_size_mode must be 'global' or 'per_device'")
    if mode == "global":
        if batch_size % world_size:
            raise ValueError(
                f"Global batch size {batch_size} must be divisible by world size {world_size}. "
                "Choose a divisible global batch or use -batch_size_mode per_device; "
                "SUPREME will not silently round or increase the global batch."
            )
        local = batch_size // world_size
    else:
        local = batch_size
    return {
        "batch_size_mode": mode,
        "requested_batch_size": batch_size,
        "per_device_batch_size": local,
        "global_microbatch_size": local * world_size,
        "data_parallel_world_size": world_size,
    }


def configure_batch_size_mode(mode):
    """Apply a CLI override to this stage without changing other stages."""
    if mode is not None:
        evaluation = os.getenv("PERFORM_EVALUATION", "false").lower() == "true"
        os.environ[
            "EVALUATION_BATCH_SIZE_MODE" if evaluation else "BATCH_SIZE_MODE"
        ] = mode


def training_batch_namespace():
    """Keep scalable runs separate from paper-compatible checkpoint caches."""
    mode = os.getenv("BATCH_SIZE_MODE", "global")
    if mode not in ("global", "per_device"):
        raise ValueError("BATCH_SIZE_MODE must be global or per_device")
    return "batch_per_device" if mode == "per_device" else ""
