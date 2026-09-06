"""ZRF (Zero Retrain Forgetting) evaluation metric.

Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" (https://arxiv.org/abs/2205.08096)
Reference: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/metrics.py#L10
"""

from torch.nn import functional as F
import torch
from supreme.eval_metrics.jsdiv import js_divergence_elements
from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_rank_values,
)
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric


@track_evaluation_metric
def ZRF(
    fabric,
    model1,
    model2,
    test_dataloader,
    metric_name="zrf",
    do_global_aggregation=True,
):
    # Preserve SUPREME's existing divergence definition, but accumulate its
    # sufficient statistics on-device instead of retaining dataset predictions.
    local_divergence_sum = torch.zeros((), dtype=torch.float64, device=fabric.device)
    local_element_count = torch.zeros_like(local_divergence_sum)
    local_offset = 0
    with torch.no_grad():
        for x, _, _ in test_dataloader:
            p = F.softmax(model1(x).float(), dim=1)
            q = F.softmax(model2(x).float(), dim=1)
            valid = evaluation_valid_mask(
                fabric, test_dataloader, local_offset, len(p), p.device
            )
            local_offset += len(p)
            contributions = js_divergence_elements(p, q)[valid]
            local_divergence_sum += contributions.double().sum()
            local_element_count += contributions.numel()
    local_zrf = 1 - local_divergence_sum / local_element_count

    if do_global_aggregation:
        gathered_stats = gather_rank_values(
            fabric,
            torch.stack([local_divergence_sum, local_element_count]),
        )
        total_stats = gathered_stats.sum(dim=0)
        if total_stats[1].item() == 0:
            raise ValueError("Cannot calculate ZRF on an empty dataset")
        final_zrf = (1 - total_stats[0] / total_stats[1]).item()
        gathered_zrf = torch.where(
            gathered_stats[:, 1] > 0,
            1 - gathered_stats[:, 0] / gathered_stats[:, 1],
            torch.nan,
        )
        # fabric.print("Final ZRF is calculated successfully")
    else:
        if local_element_count.item() == 0:
            raise ValueError("Cannot calculate ZRF on an empty dataset")
        final_zrf = (
            local_zrf.item() if isinstance(local_zrf, torch.Tensor) else local_zrf
        )
        gathered_zrf = torch.tensor([final_zrf])  # For consistent logging format

    # # Track epoch end with final ZRF value
    # ZRF.track_epoch_end(fabric, 0, final_zrf)

    zrf_dict = {
        "final_value": final_zrf,
        "per_process": gathered_zrf.cpu().tolist(),
    }

    return zrf_dict
