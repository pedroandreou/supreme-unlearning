"""Membership Inference Attack (MIA) evaluation metric.

Paper: "Selective Forgetting" (https://arxiv.org/abs/2308.07707)
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/metrics.py#L78
"""

import torch
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F

from supreme.eval_metrics.distributed import (
    evaluation_valid_mask,
    gather_masked_rows,
)
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_entropy(fabric, dataloader, model):
    """Batched, shard-local inference; retain one entropy scalar per sample."""
    values, masks = [], []
    local_offset = 0
    with torch.no_grad():
        for data, _, _ in dataloader:
            batch_entropy = entropy(F.softmax(model(data).float(), dim=-1))
            values.append(batch_entropy)
            masks.append(
                evaluation_valid_mask(
                    fabric,
                    dataloader,
                    local_offset,
                    len(batch_entropy),
                    batch_entropy.device,
                )
            )
            local_offset += len(batch_entropy)
    if not values:
        return (
            torch.empty(0, device=fabric.device),
            torch.empty(0, dtype=torch.bool, device=fabric.device),
        )
    return torch.cat(values), torch.cat(masks)


def get_membership_attack_data(
    fabric,
    num_gpus,
    retain_dataloader,
    forget_dataloader,
    test_dataloader,
    model,
    do_global_aggregation,
):
    features = []
    for loader in (retain_dataloader, forget_dataloader, test_dataloader):
        values, valid = collect_entropy(fabric, loader, model)
        if do_global_aggregation:
            values = gather_masked_rows(fabric, values, valid)
        else:
            values = values[valid]
        if values.numel() == 0:
            raise ValueError("MIA requires nonempty retain, forget and test datasets")
        features.append(values)
    retain, forget, test = features
    X_r = torch.cat([retain, test]).reshape(-1, 1)
    Y_r = torch.cat([torch.ones_like(retain), torch.zeros_like(test)])
    return forget.reshape(-1, 1), X_r, Y_r


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
    failure = None
    if fabric.global_rank == 0 or not do_global_aggregation:
        try:
            clf = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
                random_state=0,
            )
            clf.fit(X_r.cpu().numpy().reshape(-1, 1), Y_r.cpu().numpy().reshape(-1))
            predictions = clf.predict(X_f.cpu().numpy().reshape(-1, 1))
            final_mean.fill_(float(predictions.mean()))
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"

    # Keep sklearn on rank zero and communicate only the small scalar result.
    if do_global_aggregation:
        failure = fabric.broadcast(failure, src=0)
    if failure is not None:
        raise RuntimeError(f"MIA classifier failed: {failure}")
    if do_global_aggregation:
        final_mean = fabric.broadcast(final_mean, src=0)
    final_value = final_mean.item()

    return {
        "final_value": final_value,
        "per_process": [final_value] * fabric.world_size,
    }
