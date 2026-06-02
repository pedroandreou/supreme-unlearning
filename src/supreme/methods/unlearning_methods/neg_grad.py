"""NegGrad unlearning method.

Fine-tunes on the forget set Df by moving in the direction of increasing loss,
i.e. using a negative gradient for the samples to forget. This aims to damage the
features that predict Df correctly.

Paper: "Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks" (https://arxiv.org/abs/1911.04933)
Reference: https://github.com/kklusd/Unlearning/blob/main/mu/mu_basic.py#L36

Notes:
No implementation was found in the authors' repository
(https://github.com/AdityaGolatkar/SelectiveForgetting); the reference above is a
third-party implementation.
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
