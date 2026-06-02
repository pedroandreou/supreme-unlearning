"""Membership Inference Attack (MIA) evaluation metric.

Paper: "Selective Forgetting" (https://arxiv.org/abs/2308.07707)
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/metrics.py#L78

Notes:
Other approaches to Membership Inference Attack that were considered:

- "Amnesiac Machine Learning" (https://arxiv.org/abs/2010.10981)
  https://github.com/lmgraves/AmnesiacML/blob/main/Membership%20Inference%20Attack.ipynb

- "Evaluating Machine Unlearning via Epistemic Uncertainty" (https://arxiv.org/abs/2208.10836)
  https://github.com/ROYALBEFF/evaluating_machine_unlearning_via_epistemic_uncertainty/blob/735c75bb3834b3c63e4cb21f53cfce94262b88da/attack.py#L15

  or
  https://github.com/kklusd/Unlearning/blob/f14d800209c1fcf2ce9d53a707652eef083abb92/salUN/evaluation/MIA.py#L175
  https://github.com/kklusd/Unlearning/blob/f14d800209c1fcf2ce9d53a707652eef083abb92/salUN/evaluation/SVC_MIA.py

  or
  https://github.com/vlgiitr/Machine_Unlearning/blob/main/evaluation/MIA.py
  https://github.com/vlgiitr/Machine_Unlearning/blob/main/evaluation/SVC_MIA.py

  or
  https://github.com/ayushkumartarun/deep-regression-unlearning/blob/main/AgeDB_AllCNN_0to30_forgetting.ipynb
"""

from sklearn.linear_model import LogisticRegression

# from sklearn.svm import SVC
import numpy as np
import torch
from torch.nn import functional as F
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric
from supreme.utils.generic_utils import create_dataloader
import os


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(fabric, num_gpus, dataloader, model):
    # Create the dataloader with appropriate settings
    temp_dataloader = create_dataloader(
        dataset=dataloader.dataset,
        batch_size=1,
        is_training=False,  # This sets shuffle=False
        num_workers=32 if not os.environ.get("SCALENE") else 0,
        num_gpus=num_gpus,
    )

    temp_dataloader = fabric.setup_dataloaders(temp_dataloader)

    prob = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(temp_dataloader):
            # get_membership_attack_prob.track_batch_start(fabric)

            # batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, _, target = batch
            output = model(data)

            batch_prob = F.softmax(output, dim=-1).data
            prob.append(batch_prob)

            # get_membership_attack_prob.track_batch_end(
            #    fabric, batch_idx, 0, batch_prob.mean().item()
            # )

    # CRITICAL: Explicitly delete the temporary Fabric-wrapped dataloader
    # to prevent segfault during Python's exit cleanup
    del temp_dataloader

    return torch.cat(prob)


def get_membership_attack_data(
    fabric,
    num_gpus,
    retain_dataloader,
    forget_dataloader,
    test_dataloader,
    model,
):
    retain_prob = collect_prob(
        fabric=fabric, num_gpus=num_gpus, dataloader=retain_dataloader, model=model
    )
    # fabric.print(f"probability on retain set: {retain_prob}")
    forget_prob = collect_prob(
        fabric=fabric, num_gpus=num_gpus, dataloader=forget_dataloader, model=model
    )
    # fabric.print(f"probability on forget set: {forget_prob}")
    test_prob = collect_prob(
        fabric=fabric, num_gpus=num_gpus, dataloader=test_dataloader, model=model
    )
    # fabric.print(f"probability on test set: {test_prob}")

    # Return Tensors, not numpy arrays, so they can be gathered by Fabric.
    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).reshape(-1, 1)
    # MPS does not support float64 - cast to float32 on Apple Silicon only
    _dtype = np.float32 if torch.backends.mps.is_available() else np.float64
    Y_r = torch.from_numpy(
        np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))]).astype(
            _dtype
        )
    ).to(fabric.device)

    X_f = entropy(forget_prob).reshape(-1, 1)
    Y_f = torch.from_numpy(
        np.concatenate([np.ones(len(forget_prob))]).astype(_dtype)
    ).to(fabric.device)

    return X_f, Y_f, X_r, Y_r


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
    # # Track epoch start
    # get_membership_attack_prob.track_epoch_start(fabric, 0, "membership_inference")

    # Local variables (now returns tensors)
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        fabric=fabric,
        num_gpus=num_gpus,
        retain_dataloader=retain_dataloader,
        forget_dataloader=forget_dataloader,
        test_dataloader=test_dataloader,
        model=model,
    )

    clf = None
    if fabric.global_rank == 0:
        clf = LogisticRegression(
            class_weight="balanced", solver="lbfgs", multi_class="multinomial"
        )
    fabric.barrier()

    clf = fabric.broadcast(clf, src=0)

    if do_global_aggregation:
        # Gather tensors from all processes
        X_f = fabric.all_gather(X_f)
        Y_f = fabric.all_gather(Y_f)
        X_r = fabric.all_gather(X_r)
        Y_r = fabric.all_gather(Y_r)

    # Now that tensors are gathered, convert to numpy for scikit-learn
    X_f_np = X_f.cpu().numpy().reshape(-1, 1)
    # Y_f_np = Y_f.cpu().numpy().reshape(-1)
    X_r_np = X_r.cpu().numpy().reshape(-1, 1)
    Y_r_np = Y_r.cpu().numpy().reshape(-1)

    results = None
    # The training of the classifier should only happen on the main process
    # to avoid redundant work.
    if fabric.global_rank == 0:
        clf.fit(X_r_np, Y_r_np)
        results = clf.predict(X_f_np)
    fabric.barrier()

    results = fabric.broadcast(results, src=0)

    # To get the final mean, we need to gather the results if distributed
    if do_global_aggregation:
        # The `results` numpy array needs to be a tensor to be gathered
        results_tensor = torch.from_numpy(results)
        final_mean = results_tensor.cpu().numpy().mean()
        # Create a tensor for per-process values (in this case, all processes have the same value)
        gathered_mia = torch.tensor([final_mean])
    else:
        # If not distributed, or on a worker process before broadcast
        final_mean = results.mean() if results is not None else 0.0
        gathered_mia = torch.tensor([final_mean])

    # # Track epoch end with final result
    # get_membership_attack_prob.track_epoch_end(fabric, 0, final_mean)

    # Return a dictionary matching the expected format
    mia_dict = {
        "final_value": final_mean,
        "per_process": gathered_mia.tolist(),
    }

    return mia_dict
