"""Completeness evaluation metric.

Paper: "Towards Making Systems Forget with Machine Unlearning" (https://ieeexplore.ieee.org/document/7163042)
Reference: https://github.com/theLauA/MachineUnlearningPy/blob/b59dcd1d6d028b7807a56897b3911fd6989f0c02/lenskit/algorithms/item_knn.py#L141

Notes:
The reference implementation was not very clear, so we made our own based on the
description in the paper that introduced this metric, "Towards Making Systems
Forget with Machine Unlearning"
(https://www.ieee-security.org/TC/SP2015/papers-archived/6949a463.pdf), where the
authors explicitly define Completeness as the percentage of input samples where
the models produce identical predictions.
"""

import torch
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric
from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_rank_values,
)


@track_evaluation_metric
def calculate_completeness(
    fabric, model1, model2, test_dataloader, do_global_aggregation=True
):
    """
    Calculate completeness between an unlearned model and a model trained from scratch.
    Completeness is calculated as both the percentage of input samples with identical predictions
    from both models and the Jaccard distance between the two models' predictions.

    Returns:
    - completeness_percentage: Percentage of identical predictions
    """
    # # Track epoch start
    # calculate_completeness.track_epoch_start(fabric, 0, "completeness")

    total_samples = torch.tensor(0.0, dtype=torch.float64, device=fabric.device)
    identical_predictions = torch.tensor(0.0, dtype=torch.float64, device=fabric.device)
    local_offset = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # calculate_completeness.track_batch_start(fabric)

            x, _, y = batch

            # Get predictions from both models
            outputs_model1 = model1(x)
            outputs_model2 = model2(x)

            # Convert outputs to predicted classes and move to CPU
            _, pred_model1 = torch.max(outputs_model1, 1)
            _, pred_model2 = torch.max(outputs_model2, 1)

            valid_mask = evaluation_valid_mask(
                fabric=fabric,
                dataloader=test_dataloader,
                local_offset=local_offset,
                batch_size=pred_model1.shape[0],
                device=pred_model1.device,
            )
            local_offset += pred_model1.shape[0]

            batch_identical = torch.sum((pred_model1 == pred_model2) & valid_mask)
            identical_predictions += batch_identical.double()
            total_samples += valid_mask.sum().double()

            # # Track batch end with current batch completeness
            # batch_completeness = (batch_identical / pred_model1.size(0)) * 100
            # calculate_completeness.track_batch_end(
            #     fabric, batch_idx, 0, batch_completeness
            # )

    # Gather from all processes
    if do_global_aggregation:
        gathered_stats = gather_rank_values(
            fabric,
            torch.stack([identical_predictions, total_samples]),
        )
    else:
        gathered_stats = torch.stack([identical_predictions, total_samples]).reshape(
            1, -1
        )

    total_identical_predictions = gathered_stats[:, 0].sum().item()
    global_total_samples = gathered_stats[:, 1].sum().item()

    if global_total_samples == 0:
        raise ValueError("Cannot calculate completeness on an empty dataset")
    completeness_percentage = (
        (total_identical_predictions / global_total_samples) * 100
        if global_total_samples > 0
        else 0
    )

    # # Track epoch end with final completeness percentage
    # calculate_completeness.track_epoch_end(fabric, 0, completeness_percentage)

    completeness_dict = {
        "final_value": completeness_percentage,
        "per_process_identical_predictions": gathered_stats[:, 0].cpu().tolist(),
        "per_process_total_samples": gathered_stats[:, 1].cpu().tolist(),
    }

    # Return a dictionary
    return completeness_dict
