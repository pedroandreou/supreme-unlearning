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
from typing import List


@track_evaluation_metric
def ZRF(
    fabric,
    model1,
    model2,
    test_dataloader,
    metric_name="zrf",
    do_global_aggregation=True,
):
    model1_preds_list: List[torch.Tensor] = []
    model2_preds_list: List[torch.Tensor] = []
    valid_masks_list: List[torch.Tensor] = []
    local_offset = 0

    # # Track single epoch start since this is a single-pass metric
    # ZRF.track_epoch_start(fabric, 0, metric_name)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # ZRF.track_batch_start(fabric)

            x, y, cy = batch
            model1_output = model1(x)
            model2_output = model2(x)

            model1_preds = F.softmax(model1_output, dim=1).detach().cpu()
            model2_preds = F.softmax(model2_output, dim=1).detach().cpu()

            model1_preds_list.append(model1_preds)
            model2_preds_list.append(model2_preds)
            valid_masks_list.append(
                evaluation_valid_mask(
                    fabric=fabric,
                    dataloader=test_dataloader,
                    local_offset=local_offset,
                    batch_size=model1_preds.shape[0],
                    device=model1_preds.device,
                )
            )
            local_offset += model1_preds.shape[0]

            # # Calculate batch-level ZRF for tracking
            # batch_zrf = 1 - JSDiv(  # type: ignore
            #     fabric=fabric,
            #     p=model1_preds,
            #     q=model2_preds,
            #     do_global_aggregation=False,
            # )
            # ZRF.track_batch_end(fabric, batch_idx, 0, batch_zrf)

    # Stack local predictions
    model1_preds = torch.cat(model1_preds_list, axis=0)  # type: ignore
    model2_preds = torch.cat(model2_preds_list, axis=0)  # type: ignore
    valid_mask = torch.cat(valid_masks_list, axis=0)

    # fabric.print("Predictions from models are made successfully")

    contributions = js_divergence_elements(model1_preds, model2_preds)
    valid_contributions = contributions[valid_mask]
    if valid_contributions.numel() == 0:
        local_divergence_sum = torch.tensor(0.0, dtype=torch.float64)
        local_element_count = torch.tensor(0.0, dtype=torch.float64)
        local_zrf = torch.tensor(float("nan"))
    else:
        local_divergence_sum = valid_contributions.double().sum()
        local_element_count = torch.tensor(
            valid_contributions.numel(), dtype=torch.float64
        )
        local_zrf = 1 - local_divergence_sum / local_element_count
    # fabric.print("Local ZRF is calculated successfully")

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
