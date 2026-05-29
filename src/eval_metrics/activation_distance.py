import torch
from torch.nn import functional as F
from src.utils.unlearning.evaluation_utils import track_evaluation_metric
from typing import List

"""
Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" at https://arxiv.org/abs/2205.08096
and implementation can be found at: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/metrics.py#L63
"""


@track_evaluation_metric
@torch.no_grad()
def actv_dist(fabric, model1, model2, test_dataloader, do_global_aggregation=True):
    # # Track epoch start
    # actv_dist.track_epoch_start(fabric, 0, "activation_distance")

    distances_on_cpu_list: List[
        torch.Tensor
    ] = []  # Store list of CPU tensors (each tensor is for a batch)

    for batch_idx, batch in enumerate(test_dataloader):
        # actv_dist.track_batch_start(fabric)

        x, _, _ = batch  # x is on fabric.device
        model1_out = model1(x)  # Output is on fabric.device
        model2_out = model2(x)  # Output is on fabric.device

        # Move model outputs to CPU before further processing
        model1_out_cpu = model1_out.detach().cpu()
        model2_out_cpu = model2_out.detach().cpu()

        # Perform softmax and difference calculations on CPU
        softmax_model1_out = F.softmax(model1_out_cpu, dim=1)
        softmax_model2_out = F.softmax(model2_out_cpu, dim=1)

        # Store squared differences without taking sqrt
        diff_cpu = torch.sum(
            torch.square(softmax_model1_out - softmax_model2_out),
            dim=1,  # Sum over class probabilities for each sample
        )
        distances_on_cpu_list.append(diff_cpu)

        # # Track batch end with mean distance for this batch
        # batch_mean = diff_cpu.mean().item()
        # actv_dist.track_batch_end(fabric, batch_idx, 0, batch_mean)

    local_all_distances_cpu = torch.cat(distances_on_cpu_list, dim=0)

    if do_global_aggregation:
        gathered_all_distances_tensor = fabric.all_gather(local_all_distances_cpu)
        # Take sqrt after computing mean of squared distances
        final_distance = torch.sqrt(gathered_all_distances_tensor.cpu().mean()).item()
    else:
        gathered_all_distances_tensor = local_all_distances_cpu
        final_distance = torch.sqrt(local_all_distances_cpu.mean()).item()

    # # Track epoch end with final average distance
    # actv_dist.track_epoch_end(fabric, 0, final_distance)

    activ_dist_dict = {
        "final_value": final_distance,
        "per_process": gathered_all_distances_tensor.tolist(),
    }

    return activ_dist_dict
