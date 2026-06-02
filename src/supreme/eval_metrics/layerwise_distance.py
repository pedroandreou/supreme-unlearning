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
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric


@track_evaluation_metric
@torch.no_grad()
# def lay_dist(fabric, model1, model2):
def lay_dist(fabric, start_idx, end_idx, param_pairs, do_global_aggregation=True):
    lay_dist.track_epoch_start(fabric, 0, "layerwise_distance")

    # Process assigned parameters
    local_distance = torch.tensor(0.0, device=fabric.device)

    for batch_idx, i in enumerate(range(start_idx, end_idx)):
        lay_dist.track_batch_start(fabric)

        (k, p), (k0, p0) = param_pairs[i]

        p_local = p.detach().cpu()
        p0_local = p0.detach().cpu()

        # Keep calculations on GPU if available
        current_dist = (p_local - p0_local).pow(2).sum()
        # print(f"Rank {rank}: Current distance: {current_dist}")
        local_distance += current_dist

        # # Track individual layer distance (this is per-layer, so take sqrt here)
        layer_dist = torch.sqrt(current_dist).item()
        lay_dist.track_batch_end(fabric, batch_idx, 0, layer_dist)

    # Aggregate results across all ranks
    if do_global_aggregation:
        all_distances = fabric.all_gather(local_distance)
        total_squared_distance = all_distances.sum()
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
