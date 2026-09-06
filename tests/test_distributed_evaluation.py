"""Two-process regression tests for Stage 3 aggregation and collectives."""

import math
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_masked_rows,
)
from supreme.eval_metrics.layerwise_distance import lay_dist, model_lay_dist
from supreme.eval_metrics.activation_distance import actv_dist
from supreme.eval_metrics.completeness import calculate_completeness
from supreme.eval_metrics.jsdiv import js_divergence_elements
from supreme.eval_metrics.zrf import ZRF
from supreme.eval_metrics.membership_inference_attack import get_membership_attack_prob
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
        if not isinstance(data, torch.Tensor):
            objects = [data]
            dist.broadcast_object_list(objects, src=src)
            return objects[0]
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


class _AlwaysClassOne(torch.nn.Module):
    def forward(self, inputs):
        return torch.tensor([0.0, 1.0]).repeat(inputs.shape[0], 1)


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
        empty = gather_masked_rows(
            fabric, torch.empty(0), torch.empty(0, dtype=torch.bool)
        )
        assert empty.numel() == 0

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

        # Cover both uneven padding and a rank with zero valid rows (N < P).
        for size in (1, 5):
            subset = torch.utils.data.Subset(dataset, range(size))
            sampler = DistributedSampler(
                subset, num_replicas=world_size, rank=rank, shuffle=False
            )
            loader = DataLoader(subset, sampler=sampler, batch_size=2)
            first, second = _AlwaysClassZero(), _AlwaysClassOne()
            p, q = (
                torch.tensor([[1.0, 0.0]]).softmax(1),
                torch.tensor([[0.0, 1.0]]).softmax(1),
            )
            zrf = ZRF(fabric, first, second, loader)["metric_value_dict"]["final_value"]
            assert math.isclose(
                zrf, 1 - js_divergence_elements(p, q).mean().item(), abs_tol=1e-7
            )
            distance = actv_dist(fabric, first, second, loader)["metric_value_dict"][
                "final_value"
            ]
            assert math.isclose(
                distance, (p - q).square().sum().sqrt().item(), abs_tol=1e-7
            )
            complete = calculate_completeness(fabric, first, second, loader)[
                "metric_value_dict"
            ]
            assert complete["final_value"] == 0
            assert sum(complete["per_process_total_samples"]) == size
            mia = get_membership_attack_prob(
                fabric, world_size, first, loader, loader, loader
            )["metric_value_dict"]
            assert 0 <= mia["final_value"] <= 1

        # A classifier failure on rank zero must reach every rank, not leave
        # its peers waiting for a result broadcast that rank zero never enters.
        import supreme.eval_metrics.membership_inference_attack as mia_module

        original_fit = mia_module.LogisticRegression.fit

        def fail_fit(*args, **kwargs):
            raise ValueError("injected classifier error")

        mia_module.LogisticRegression.fit = fail_fit
        try:
            with pytest.raises(RuntimeError, match="injected classifier error"):
                get_membership_attack_prob(
                    fabric, world_size, first, loader, loader, loader
                )
        finally:
            mia_module.LogisticRegression.fit = original_fit

        torch.manual_seed(12)
        first, second = torch.nn.Linear(100, 2), torch.nn.Linear(100, 2)
        expected = (
            sum(
                (p.double() - q.double()).square().sum()
                for p, q in zip(first.parameters(), second.parameters())
            )
            .sqrt()
            .item()
        )
        actual = model_lay_dist(fabric, first, second)["metric_value_dict"][
            "final_value"
        ]
        assert math.isclose(actual, expected, abs_tol=1e-7)
        assert (
            model_lay_dist(fabric, first, first)["metric_value_dict"]["final_value"]
            == 0
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
def test_stage3_multi_process_aggregation(tmp_path, world_size):
    init_method = f"file://{tmp_path / 'distributed-evaluation-init'}"
    mp.spawn(
        _distributed_evaluation_worker,
        args=(world_size, init_method),
        nprocs=world_size,
        join=True,
    )
