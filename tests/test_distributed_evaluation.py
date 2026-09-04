"""Two-process regression tests for Stage 3 aggregation and collectives."""

import math
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_masked_rows,
)
from supreme.eval_metrics.layerwise_distance import lay_dist
from supreme.utils.training.training_utils import evaluate


class _DistributedFabricStub:
    """Small Fabric-compatible facade backed by a real Gloo process group."""

    device = torch.device("cpu")

    @property
    def global_rank(self):
        return dist.get_rank()

    @property
    def world_size(self):
        return dist.get_world_size()

    def all_gather(self, data):
        gathered = [torch.empty_like(data) for _ in range(self.world_size)]
        dist.all_gather(gathered, data)
        return torch.stack(gathered)

    def broadcast(self, data, src=0):
        dist.broadcast(data, src=src)
        return data

    def barrier(self):
        dist.barrier()

    def call(self, *args, **kwargs):
        pass

    def print(self, *args, **kwargs):
        pass


class _AlwaysClassZero(torch.nn.Module):
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        return torch.tensor([1.0, 0.0]).repeat(batch_size, 1)


def _distributed_evaluation_worker(rank, world_size, init_method):
    loopback_interfaces = {
        name for _, name in socket.if_nameindex() if name in {"lo", "lo0"}
    }
    if loopback_interfaces:
        os.environ["GLOO_SOCKET_IFNAME"] = sorted(loopback_interfaces)[0]
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    try:
        fabric = _DistributedFabricStub()
        labels = torch.tensor([0, 1, 0, 1, 0])
        dataset = TensorDataset(torch.arange(5).float().reshape(-1, 1), labels, labels)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

        local_values = []
        local_valid = []
        local_offset = 0
        for inputs, _, _ in dataloader:
            local_values.append(inputs)
            local_valid.append(
                evaluation_valid_mask(
                    fabric,
                    dataloader,
                    local_offset,
                    inputs.shape[0],
                    inputs.device,
                )
            )
            local_offset += inputs.shape[0]

        gathered = gather_masked_rows(
            fabric,
            torch.cat(local_values),
            torch.cat(local_valid),
        )
        assert sorted(gathered.flatten().tolist()) == [0.0, 1.0, 2.0, 3.0, 4.0]

        # The sampler pads index 0 to six rows. A naive calculation gives
        # 4/6, while padding-aware aggregation must return the true 3/5.
        evaluation = evaluate(
            fabric=fabric,
            model=_AlwaysClassZero(),
            test_dataloader=dataloader,
            do_global_aggregation=True,
        )
        assert evaluation["metric_value_dict"]["Acc"]["final_value"] == 60.0

        parameters = [
            (
                (f"p{i}", torch.nn.Parameter(torch.tensor([float(i)]))),
                (f"p{i}", torch.nn.Parameter(torch.tensor([0.0]))),
            )
            for i in (1, 2, 3)
        ]
        params_per_rank = len(parameters) // world_size
        start_idx = rank * params_per_rank
        end_idx = (
            start_idx + params_per_rank if rank < world_size - 1 else len(parameters)
        )
        distance = lay_dist(
            fabric=fabric,
            start_idx=start_idx,
            end_idx=end_idx,
            param_pairs=parameters,
            do_global_aggregation=True,
        )
        assert math.isclose(
            distance["metric_value_dict"]["final_value"],
            math.sqrt(14.0),
            rel_tol=1e-6,
        )
    finally:
        dist.destroy_process_group()


def test_stage3_two_process_aggregation(tmp_path):
    init_method = f"file://{tmp_path / 'distributed-evaluation-init'}"
    mp.spawn(
        _distributed_evaluation_worker,
        args=(2, init_method),
        nprocs=2,
        join=True,
    )
