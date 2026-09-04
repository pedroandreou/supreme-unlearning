"""Distributed helpers shared by SUPREME evaluation metrics.

Evaluation keeps Fabric's padded ``DistributedSampler`` so every rank executes
the same number of forward passes.  This is important for DDP and FSDP, whose
forward paths can contain collectives.  The helpers below identify sampler
padding and remove its contribution from metric totals.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler


def gather_rank_values(fabric, values: Tensor) -> Tensor:
    """All-gather a fixed-size vector and return ``[world_size, fields]``.

    Fabric 2.1 does not add a world dimension when ``world_size == 1``.  The
    reshape normalises that special case and gives callers one representation
    for both single-process and distributed execution.
    """

    local_values = values.detach().to(device=fabric.device).reshape(-1)
    gathered = fabric.all_gather(local_values)
    return gathered.reshape(fabric.world_size, -1)


def evaluation_valid_mask(
    fabric,
    dataloader,
    local_offset: int,
    batch_size: int,
    device: Union[str, torch.device],
) -> Tensor:
    """Return a mask that excludes padding added by ``DistributedSampler``.

    With ``drop_last=False``, PyTorch appends padding indices to a global index
    list and then gives rank ``r`` positions ``r, r + P, r + 2P, ...``.  A
    local item is therefore real exactly when its position in that global list
    is smaller than the dataset length.  This remains true when the sampler
    shuffles the underlying indices.
    """

    if fabric.world_size == 1:
        return torch.ones(batch_size, dtype=torch.bool, device=device)

    sampler = getattr(dataloader, "sampler", None)
    if not isinstance(sampler, DistributedSampler):
        raise RuntimeError(
            "Distributed evaluation requires a torch DistributedSampler so "
            "that each rank evaluates a distinct data shard."
        )
    if sampler.drop_last:
        raise RuntimeError(
            "Distributed evaluation requires drop_last=False; otherwise some "
            "evaluation samples are omitted."
        )
    if sampler.num_replicas != fabric.world_size or sampler.rank != fabric.global_rank:
        raise RuntimeError(
            "Evaluation sampler rank/world size does not match the Fabric process group."
        )

    local_positions = torch.arange(
        local_offset,
        local_offset + batch_size,
        dtype=torch.long,
        device=device,
    )
    global_positions = fabric.global_rank + local_positions * fabric.world_size
    return global_positions < len(sampler.dataset)


def gather_masked_rows(fabric, local_values: Tensor, local_valid: Tensor) -> Tensor:
    """All-gather variable first-dimension tensors and discard invalid rows.

    The current evaluation loaders have equal local lengths because their
    distributed samplers pad.  Length gathering and explicit tensor padding
    make the helper safe for custom non-padding samplers as well.
    """

    local_values = local_values.detach().to(fabric.device)
    local_valid = local_valid.detach().to(device=fabric.device, dtype=torch.bool)

    if local_values.ndim == 0:
        local_values = local_values.reshape(1)
    if local_values.shape[0] != local_valid.numel():
        raise ValueError("local_values and local_valid must have the same row count")

    local_length = torch.tensor(
        [local_values.shape[0]], dtype=torch.long, device=fabric.device
    )
    gathered_lengths = gather_rank_values(fabric, local_length).reshape(-1).long()
    max_length = int(gathered_lengths.max().item())

    if local_values.shape[0] < max_length:
        pad_shape = (max_length - local_values.shape[0], *local_values.shape[1:])
        local_values = torch.cat(
            [
                local_values,
                torch.zeros(
                    pad_shape,
                    dtype=local_values.dtype,
                    device=local_values.device,
                ),
            ],
            dim=0,
        )
        local_valid = torch.cat(
            [
                local_valid,
                torch.zeros(
                    max_length - local_valid.numel(),
                    dtype=torch.bool,
                    device=local_valid.device,
                ),
            ],
            dim=0,
        )

    gathered_values = fabric.all_gather(local_values)
    gathered_valid = fabric.all_gather(local_valid.to(torch.uint8)).bool()

    if fabric.world_size == 1:
        gathered_values = gathered_values.unsqueeze(0)
        gathered_valid = gathered_valid.unsqueeze(0)

    rows = []
    for rank, rank_length in enumerate(gathered_lengths.tolist()):
        rank_values = gathered_values[rank, :rank_length]
        rank_valid = gathered_valid[rank, :rank_length]
        rows.append(rank_values[rank_valid])

    if not rows:
        return local_values[:0]
    return torch.cat(rows, dim=0)
