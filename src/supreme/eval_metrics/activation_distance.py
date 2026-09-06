"""Activation Distance evaluation metric.

Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" (https://arxiv.org/abs/2205.08096)
Reference: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/metrics.py#L63
"""

import torch
from torch.nn import functional as F
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric
from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_rank_values,
)


@track_evaluation_metric
@torch.no_grad()
def actv_dist(fabric, model1, model2, test_dataloader, do_global_aggregation=True):
    # # Track epoch start
    # actv_dist.track_epoch_start(fabric, 0, "activation_distance")

    local_squared_distance = torch.tensor(
        0.0, dtype=torch.float64, device=fabric.device
    )
    local_sample_count = torch.tensor(0.0, dtype=torch.float64, device=fabric.device)
    local_offset = 0

    for batch_idx, batch in enumerate(test_dataloader):
        # actv_dist.track_batch_start(fabric)

        x, _, _ = batch  # x is on fabric.device
        model1_out = model1(x)  # Output is on fabric.device
        model2_out = model2(x)  # Output is on fabric.device

        # Compute probabilities and squared differences on each rank's device.
        model1_out_cpu = model1_out.detach().float()
        model2_out_cpu = model2_out.detach().float()

        # Float32 metric arithmetic also avoids low-precision underflow.
        softmax_model1_out = F.softmax(model1_out_cpu, dim=1)
        softmax_model2_out = F.softmax(model2_out_cpu, dim=1)

        # Store squared differences without taking sqrt
        diff_cpu = torch.sum(
            torch.square(softmax_model1_out - softmax_model2_out),
            dim=1,  # Sum over class probabilities for each sample
        )
        valid_mask = evaluation_valid_mask(
            fabric=fabric,
            dataloader=test_dataloader,
            local_offset=local_offset,
            batch_size=diff_cpu.shape[0],
            device=diff_cpu.device,
        )
        local_offset += diff_cpu.shape[0]
        local_squared_distance += diff_cpu[valid_mask].double().sum()
        local_sample_count += valid_mask.sum().double()

        # # Track batch end with mean distance for this batch
        # batch_mean = diff_cpu.mean().item()
        # actv_dist.track_batch_end(fabric, batch_idx, 0, batch_mean)

    if do_global_aggregation:
        gathered_stats = gather_rank_values(
            fabric,
            torch.stack([local_squared_distance, local_sample_count]),
        )
    else:
        gathered_stats = torch.stack(
            [local_squared_distance, local_sample_count]
        ).reshape(1, -1)

    total_stats = gathered_stats.sum(dim=0)
    if total_stats[1].item() == 0:
        raise ValueError("Cannot calculate activation distance on an empty dataset")
    final_distance = torch.sqrt(total_stats[0] / total_stats[1]).item()
    per_process = torch.where(
        gathered_stats[:, 1] > 0,
        torch.sqrt(gathered_stats[:, 0] / gathered_stats[:, 1]),
        torch.nan,
    )

    # # Track epoch end with final average distance
    # actv_dist.track_epoch_end(fabric, 0, final_distance)

    activ_dist_dict = {
        "final_value": final_distance,
        "per_process": per_process.cpu().tolist(),
    }

    return activ_dist_dict
