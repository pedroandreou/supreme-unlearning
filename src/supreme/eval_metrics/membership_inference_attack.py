"""Membership Inference Attack (MIA) evaluation metric.

Paper: "Selective Forgetting" (https://arxiv.org/abs/2308.07707)
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/metrics.py#L78
"""

import os

import torch
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_masked_rows,
)
from supreme.utils.generic_utils import create_dataloader
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(fabric, num_gpus, dataloader, model):
    """Collect local probabilities and mark sampler-padding rows."""
    temp_dataloader = create_dataloader(
        dataset=dataloader.dataset,
        batch_size=1,
        is_training=False,
        num_workers=32 if not os.environ.get("SCALENE") else 0,
        num_gpus=num_gpus,
    )
    temp_dataloader = fabric.setup_dataloaders(temp_dataloader)

    probabilities = []
    valid_rows = []
    local_offset = 0

    with torch.no_grad():
        for batch in temp_dataloader:
            data, _, target = batch
            batch_prob = F.softmax(model(data), dim=-1)
            probabilities.append(batch_prob)
            valid_rows.append(
                evaluation_valid_mask(
                    fabric,
                    temp_dataloader,
                    local_offset,
                    batch_prob.shape[0],
                    batch_prob.device,
                )
            )
            local_offset += batch_prob.shape[0]

    # Avoid keeping the temporary Fabric-wrapped dataloader alive during exit.
    del temp_dataloader

    return torch.cat(probabilities), torch.cat(valid_rows)


def get_membership_attack_data(
    fabric,
    num_gpus,
    retain_dataloader,
    forget_dataloader,
    test_dataloader,
    model,
    do_global_aggregation,
):
    retain_prob, retain_valid = collect_prob(
        fabric=fabric, num_gpus=num_gpus, dataloader=retain_dataloader, model=model
    )
    forget_prob, forget_valid = collect_prob(
        fabric=fabric, num_gpus=num_gpus, dataloader=forget_dataloader, model=model
    )
    test_prob, test_valid = collect_prob(
        fabric=fabric, num_gpus=num_gpus, dataloader=test_dataloader, model=model
    )

    if do_global_aggregation:
        retain_prob = gather_masked_rows(fabric, retain_prob, retain_valid)
        forget_prob = gather_masked_rows(fabric, forget_prob, forget_valid)
        test_prob = gather_masked_rows(fabric, test_prob, test_valid)
    else:
        retain_prob = retain_prob[retain_valid]
        forget_prob = forget_prob[forget_valid]
        test_prob = test_prob[test_valid]

    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).reshape(-1, 1)
    Y_r = torch.cat(
        [
            torch.ones(len(retain_prob), device=X_r.device),
            torch.zeros(len(test_prob), device=X_r.device),
        ]
    )
    X_f = entropy(forget_prob).reshape(-1, 1)

    return X_f, X_r, Y_r


@track_evaluation_metric
def get_membership_attack_prob(
    fabric,
    num_gpus,
    model,
    retain_dataloader,
    forget_dataloader,
    test_dataloader,
    do_global_aggregation=True,
):
    X_f, X_r, Y_r = get_membership_attack_data(
        fabric=fabric,
        num_gpus=num_gpus,
        retain_dataloader=retain_dataloader,
        forget_dataloader=forget_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        do_global_aggregation=do_global_aggregation,
    )

    final_mean = torch.zeros((), dtype=torch.float64, device=fabric.device)
    if fabric.global_rank == 0:
        clf = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
            random_state=0,
        )
        clf.fit(X_r.cpu().numpy().reshape(-1, 1), Y_r.cpu().numpy().reshape(-1))
        predictions = clf.predict(X_f.cpu().numpy().reshape(-1, 1))
        final_mean.fill_(float(predictions.mean()))

    # Keep sklearn on rank zero and communicate only the small scalar result.
    final_mean = fabric.broadcast(final_mean, src=0)
    final_value = final_mean.item()

    return {
        "final_value": final_value,
        "per_process": [final_value] * fabric.world_size,
    }
