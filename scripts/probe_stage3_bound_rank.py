"""Simulate SLURM's one-visible-GPU-per-task binding under torchrun.

This exercises SUPREME's custom SLURM environment and sampler handling on real
GPUs. It does not submit a SLURM job or establish multi-node correctness.
"""

import os
from pathlib import Path
import runpy


def configure_bound_rank():
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if (
        local_world < 1
        or world < 1
        or world % local_world
        or not 0 <= rank < world
        or not 0 <= local_rank < local_world
        or len(devices) < local_world
    ):
        raise ValueError(
            "Expected homogeneous torchrun workers with explicit visible GPUs"
        )
    os.environ.update(
        CUDA_VISIBLE_DEVICES=devices[local_rank],
        SLURM_JOB_ID="999999",
        SLURM_JOB_NAME="supreme-bound-rank-probe",
        SLURM_NTASKS=str(world),
        SLURM_NTASKS_PER_NODE=str(local_world),
        SLURM_NNODES=str(world // local_world),
        SLURM_NODEID=str(rank // local_world),
        SLURM_PROCID=str(rank),
        SLURM_LOCALID=str(local_rank),
    )


if __name__ == "__main__":
    # Must run before importing torch so CUDA observes the per-task visibility.
    configure_bound_rank()
    runpy.run_path(
        str(Path(__file__).with_name("probe_stage3_distributed.py")),
        run_name="__main__",
    )
