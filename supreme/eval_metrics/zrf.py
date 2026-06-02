from torch.nn import functional as F
import torch
from supreme.eval_metrics.jsdiv import (
    JSDiv,
)
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric
from typing import List

"""
Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" at https://arxiv.org/abs/2205.08096
and implementation can be found at: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/metrics.py#L10
"""


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

    # fabric.print("Predictions from models are made successfully")

    # Calculate local JSDiv
    local_zrf = 1 - JSDiv(  # type: ignore
        fabric=fabric, p=model1_preds, q=model2_preds, do_global_aggregation=False
    )
    # fabric.print("Local ZRF is calculated successfully")

    if do_global_aggregation:
        # Gather ZRF values from all processes and average them
        gathered_zrf = fabric.all_gather(local_zrf)
        # fabric.print("ZRF values are gathered successfully")
        final_zrf = gathered_zrf.mean().item()
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
        "per_process": gathered_zrf.tolist(),
    }

    return zrf_dict
