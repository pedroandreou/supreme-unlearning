from src.utils.training.training_utils import fit_one_learning_cycle
from lightning.fabric import Fabric


# Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" at https://arxiv.org/abs/2308.07707
# https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/forget_random_strategies.py#L110
def finetune(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: Fabric,  # Independent copy of the original model so it can go under the unlearning procedure
    retain_train_dataloader: Fabric,
    retain_test_dataloader: Fabric,
    **kwargs,
):
    # Finetune the model using the retain data for a set number of epochs.
    # Capture the NEW wrapped model (see retrain.py comment for the FSDP rationale).
    model, _ = fit_one_learning_cycle(
        fabric=fabric,
        num_gpus=num_gpus,
        epochs=5,
        model=model,
        train_dataloader=retain_train_dataloader,
        test_dataloader=retain_test_dataloader,
        lr=0.02,
        milestones=None,
        wandb_logging_flag=wandb_logging_flag,
        neggrad_flag=False,
    )

    return model
