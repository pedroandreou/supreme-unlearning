"""JS-Divergence evaluation metric.

Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" (https://arxiv.org/abs/2205.08096)
Reference: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/CIFARSuper20_Rocket_Unlearn.ipynb

Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" (https://arxiv.org/abs/2308.07707)
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/metrics.py#L12
"""

import torch
from torch.nn import functional as F
from supreme.eval_metrics.distributed import gather_rank_values


def js_divergence_elements(p, q):
    """Return element-wise divergence contributions used by SUPREME's metric."""

    m = (p + q) / 2
    kl_div_p = F.kl_div(torch.log(p), m, reduction="none")
    kl_div_q = F.kl_div(torch.log(q), m, reduction="none")
    return 0.5 * (kl_div_p + kl_div_q)


def JSDiv(fabric, p, q, do_global_aggregation=False):
    """
    p is M(x) (the unlearned model's predictions)
    q is Td(x) (the retrained or randomly initialized model's predictions)
    m is the average of p and q, just as in the mathematical formula m = (M(x)+Td(x)) / 2
    """
    # fabric.print("JSDiv is about to be calculated")
    contributions = js_divergence_elements(p, q)
    local_js_div = contributions.mean()

    final_js_div = None
    if do_global_aggregation:
        gathered_stats = gather_rank_values(
            fabric,
            torch.stack(
                [
                    contributions.double().sum(),
                    torch.tensor(
                        contributions.numel(),
                        dtype=torch.float64,
                        device=contributions.device,
                    ),
                ]
            ),
        )
        total_stats = gathered_stats.sum(dim=0)
        final_js_div = (total_stats[0] / total_stats[1]).item()
        gathered_js_div = gathered_stats[:, 0] / gathered_stats[:, 1]
        # fabric.print("JSDiv is calculated successfully")

        jsdiv_dict = {
            "final_value": final_js_div,
            "per_process": gathered_js_div.cpu().tolist(),
        }

        return jsdiv_dict
    else:
        final_js_div = local_js_div.item()
        # fabric.print("JSDiv is calculated successfully")

        return final_js_div
