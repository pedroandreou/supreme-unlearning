"""Batch policies and launcher sizing, including simulated large world sizes."""

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from supreme.utils.batching import (
    configure_batch_size_mode,
    launched_world_size,
    resolve_batch_size,
    training_batch_namespace,
)
from supreme.utils.generic_utils import create_dataloader
from supreme.eval_metrics.distributed import evaluation_valid_mask


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16, 128])
def test_global_batch_is_exact(world):
    config = resolve_batch_size(128, world, "global")
    assert config["per_device_batch_size"] == 128 // world
    assert config["global_microbatch_size"] == 128


@pytest.mark.parametrize("world", [1, 2, 3, 4, 8, 1000, 1024, 4096])
def test_per_device_batch_does_not_shrink(world):
    config = resolve_batch_size(128, world, "per_device")
    assert config["per_device_batch_size"] == 128
    assert config["global_microbatch_size"] == 128 * world


@pytest.mark.parametrize("batch,world", [(64, 3), (128, 1000), (1, 2)])
def test_global_batch_never_silently_rounds_or_clamps(batch, world):
    with pytest.raises(ValueError, match="divisible"):
        resolve_batch_size(batch, world, "global")


def test_stage_modes_are_independent(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE_MODE", "global")
    monkeypatch.setenv("EVALUATION_BATCH_SIZE_MODE", "per_device")
    monkeypatch.setenv("PERFORM_EVALUATION", "false")
    assert resolve_batch_size(64, 4)["per_device_batch_size"] == 16
    configure_batch_size_mode("per_device")
    assert resolve_batch_size(64, 4)["per_device_batch_size"] == 64
    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    configure_batch_size_mode("global")
    assert resolve_batch_size(64, 4)["per_device_batch_size"] == 16
    assert os.environ["BATCH_SIZE_MODE"] == "per_device"


def test_external_world_size_overrides_local_gpu_count(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    assert launched_world_size(4) == 16
    from supreme.utils.fabric.fabric_setup import get_slurm_node_count

    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.delenv("SLURM_JOB_NUM_NODES", raising=False)
    assert get_slurm_node_count() == 4
    monkeypatch.setenv("PERFORM_EVALUATION", "false")
    monkeypatch.setenv("BATCH_SIZE_MODE", "global")
    loader = create_dataloader(
        TensorDataset(torch.arange(100)), 64, num_gpus=4, num_workers=0
    )
    assert loader.batch_size == 4
    assert loader.supreme_batch_config["data_parallel_world_size"] == 16


def test_live_process_group_takes_precedence(monkeypatch):
    import supreme.utils.batching as batching

    monkeypatch.setattr(batching.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(batching.dist, "get_world_size", lambda: 8)
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert launched_world_size(1) == 8


def test_namespace_preserves_paper_paths(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE_MODE", "global")
    assert training_batch_namespace() == ""
    monkeypatch.setenv("BATCH_SIZE_MODE", "per_device")
    assert training_batch_namespace() == "batch_per_device"


def test_worker_override(monkeypatch):
    monkeypatch.setenv("DATALOADER_NUM_WORKERS", "0")
    loader = create_dataloader(TensorDataset(torch.arange(100)), 64)
    assert loader.num_workers == 0


@pytest.mark.parametrize("samples", [1, 500, 1025])
def test_simulated_thousand_rank_sampler_counts_each_sample_once(samples):
    # Arithmetic/sampler coverage only. This does not launch 1024 processes.
    world = 1024
    dataset = TensorDataset(torch.arange(samples))
    counts = torch.zeros(samples, dtype=torch.long)
    for rank in range(world):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=False
        )
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler)
        indices = torch.tensor(list(sampler))
        fabric = SimpleNamespace(world_size=world, global_rank=rank)
        valid = evaluation_valid_mask(fabric, loader, 0, len(indices), "cpu")
        counts.index_add_(0, indices[valid], torch.ones_like(indices[valid]))
    assert torch.all(counts == 1)


def test_slurm_large_world_dry_run_does_not_submit():
    # This validates argument/resource calculations, not 1024-GPU execution.
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash unavailable")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, SLURM_ACCOUNT="test-only")
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run(
        [
            bash,
            "src/supreme/run_slurm.sh",
            "--dry-run",
            "--nodes",
            "128",
            "--gpus-per-node",
            "8",
            "--batch-size-mode",
            "per_device",
            "--training-seeds",
            "260",
            "--datasets",
            "Cifar10",
            "--models",
            "ResNet18",
            "--strategies",
            "random_",
            "--forget-percs",
            "0.01",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "GPUs per cell: 1024" in result.stdout
    assert "Stage 1/2 batch-size mode: per_device" in result.stdout
