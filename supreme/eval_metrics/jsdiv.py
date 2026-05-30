import torch
from torch.nn import functional as F


"""
Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" at https://arxiv.org/abs/2205.08096
and implementation can be found at: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/CIFARSuper20_Rocket_Unlearn.ipynb

Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" at https://arxiv.org/abs/2308.07707
and implementation can be found at: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/metrics.py#L12
"""


def JSDiv(fabric, p, q, do_global_aggregation=False):
    """
    p is M(x) (the unlearned model's predictions)
    q is Td(x) (the retrained or randomly initialized model's predictions)
    m is the average of p and q, just as in the mathematical formula m = (M(x)+Td(x)) / 2
    """
    # fabric.print("JSDiv is about to be calculated")
    m = (p + q) / 2
    kl_div_p = F.kl_div(torch.log(p), m)
    kl_div_q = F.kl_div(torch.log(q), m)

    # Calculate local JS divergence
    local_js_div = 0.5 * (kl_div_p + kl_div_q)

    final_js_div = None
    if do_global_aggregation:
        # Gather results from all processes and average them
        gathered_js_div = fabric.all_gather(local_js_div)
        final_js_div = gathered_js_div.mean().item()
        # fabric.print("JSDiv is calculated successfully")

        jsdiv_dict = {
            "final_value": final_js_div,
            "per_process": gathered_js_div.tolist(),
        }

        return jsdiv_dict
    else:
        final_js_div = local_js_div.item()
        # fabric.print("JSDiv is calculated successfully")

        return final_js_div
