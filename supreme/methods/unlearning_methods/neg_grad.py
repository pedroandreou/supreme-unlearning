"""
Denotes fine-tuning on Df by moving in the direction of increasing loss
which is equivalent to using a negative gradient for the samples to forget.
This aims to damage features predicting Df correctly.

The method is used in Paper: "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks" at https://arxiv.org/abs/1911.04933
However, I could not find its implementation in https://github.com/AdityaGolatkar/SelectiveForgetting

But I have found an implementation of it at: https://github.com/kklusd/Unlearning/blob/main/mu/mu_basic.py#L36
"""

from supreme.utils.training.training_utils import fit_one_learning_cycle
from lightning.fabric import Fabric


def neg_grad(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: Fabric,  # Independent copy of the original model so it can go under the unlearning procedure
    forget_train_dataloader: Fabric,
    forget_test_dataloader: Fabric,
    **kwargs,
):
    # Capture the NEW wrapped model (see retrain.py comment for FSDP rationale).
    model, _ = fit_one_learning_cycle(
        fabric=fabric,
        num_gpus=num_gpus,
        epochs=5,
        model=model,
        train_dataloader=forget_train_dataloader,
        test_dataloader=forget_test_dataloader,
        lr=0.02,
        milestones=None,
        wandb_logging_flag=wandb_logging_flag,
        neggrad_flag=True,
    )

    return model
