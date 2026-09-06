"""Layer-wise Distance evaluation metric.

Also referred to as Layer-wise distance, Weight Distance, or Layer-wise Weight
Difference in "Fast Yet Effective Machine Unlearning" (https://arxiv.org/abs/2111.08947).

Paper: "DeltaGrad: Rapid retraining of machine learning models" (https://proceedings.mlr.press/v119/wu20b.html)
Reference: https://github.com/wuyinjun-1993/DeltaGrad/blob/ebb85816ba9ff6cd13dc88361886d4eae1bd7e77/src/utils.py#L447

Paper: "Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations" (https://arxiv.org/abs/2003.02960)
Reference: https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb

Paper: "Fast Yet Effective Machine Unlearning" (https://arxiv.org/abs/2111.08947)
Reference: https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb
"""

import torch
from contextlib import ExitStack, nullcontext
from supreme.eval_metrics.distributed import gather_rank_values
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric


def element_interval(total, rank, world_size):
    """Balance arithmetic by parameter elements, not by tensor count."""
    return total * rank // world_size, total * (rank + 1) // world_size


@track_evaluation_metric
@torch.no_grad()
def model_lay_dist(fabric, model1, model2, do_global_aggregation=True):
    """Compute L2 parameter distance with collective-safe sharded model access.

    FSDP temporarily materializes full weights (so peak memory rises). ZeRO 3
    materializes one parameter pair at a time. Every rank enters those contexts
    in the same order, but computes only its element-balanced interval.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    with ExitStack() as stack:
        seen = set()
        for model in (model1, model2):
            if id(model) not in seen and any(
                isinstance(m, FSDP) for m in model.modules()
            ):
                stack.enter_context(FSDP.summon_full_params(model, writeback=False))
            seen.add(id(model))
        first, second = dict(model1.named_parameters()), dict(model2.named_parameters())
        if first.keys() != second.keys():
            raise ValueError("Layerwise distance requires matching parameter names")
        pairs = [(first[name], second[name]) for name in first]
        sizes = [getattr(p, "ds_numel", p.numel()) for p, _ in pairs]
        for (p, q), size in zip(pairs, sizes):
            if size != getattr(q, "ds_numel", q.numel()) or getattr(
                p, "ds_shape", p.shape
            ) != getattr(q, "ds_shape", q.shape):
                raise ValueError(
                    "Layerwise distance requires matching parameter shapes"
                )
        total = sum(sizes)
        start, end = (
            element_interval(total, fabric.global_rank, fabric.world_size)
            if do_global_aggregation
            else (0, total)
        )
        local = torch.zeros((), device=fabric.device, dtype=torch.float64)
        offset = 0
        for (p, q), size in zip(pairs, sizes):
            zero_params = list(
                {id(t): t for t in (p, q) if hasattr(t, "ds_id")}.values()
            )
            context = nullcontext()
            if zero_params:
                from deepspeed import zero

                context = zero.GatheredParameters(zero_params, modifier_rank=None)
            with context:
                lo, hi = max(0, start - offset), min(size, end - offset)
                # Bound temporary difference storage even for a very large tensor.
                for chunk in range(lo, max(lo, hi), 1_048_576):
                    stop = min(chunk + 1_048_576, hi)
                    delta = (
                        p.reshape(-1)[chunk:stop].float()
                        - q.reshape(-1)[chunk:stop].float()
                    )
                    local += delta.double().square().sum()
            offset += size
    distances = (
        gather_rank_values(fabric, local.reshape(1))[:, 0]
        if do_global_aggregation
        else local.reshape(1)
    )
    return {
        "final_value": distances.sum().sqrt().item(),
        "per_process": distances.cpu().tolist(),
    }


@track_evaluation_metric
@torch.no_grad()
# def lay_dist(fabric, model1, model2):
def lay_dist(fabric, start_idx, end_idx, param_pairs, do_global_aggregation=True):
    lay_dist.track_epoch_start(fabric, 0, "layerwise_distance")

    # Process assigned parameters
    local_distance = torch.tensor(0.0, device=fabric.device)

    for i in range(start_idx, end_idx):
        (k, p), (k0, p0) = param_pairs[i]

        current_dist = (p.detach().float() - p0.detach().float()).pow(2).sum()
        local_distance += current_dist.to(fabric.device)

    # Aggregate results across all ranks
    if do_global_aggregation:
        all_distances = gather_rank_values(fabric, local_distance.reshape(1))[:, 0]
    else:
        all_distances = local_distance.unsqueeze(0)
    total_squared_distance = all_distances.sum()

    # Apply final square root as per the paper's formula
    total_distance = torch.sqrt(total_squared_distance).item()

    lay_dist.track_epoch_end(fabric, 0, total_distance)

    lay_dist_dict = {
        "final_value": total_distance,
        "per_process": all_distances.tolist(),
    }

    return lay_dist_dict
