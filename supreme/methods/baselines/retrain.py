from supreme.utils import project_config as project_config
from supreme.utils.training.training_utils import fit_one_learning_cycle
from supreme.utils.generic_utils import initialize_network
from lightning.fabric import Fabric
import torch


# Retrain the model on the retain set only
def retrain(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: torch.nn.Module,  # Independent copy of the original model so it can go under the unlearning procedure
    retain_train_dataloader: torch.utils.data.DataLoader,
    retain_test_dataloader: torch.utils.data.DataLoader,
    model_name: str,
    dataset_name: str,
    **kwargs,
):
    if model_name == "ViT":
        # ViT: re-create from HuggingFace pretrained weights for a true retrain-from-scratch.
        # The shallow reset_parameters loop only resets nn.Linear children, leaving the
        # fine-tuned ViT backbone intact which is not a fair baseline.
        raw_model = model.module if hasattr(model, "module") else model
        num_labels = kwargs.get("num_labels") or raw_model.num_labels
        device = next(model.parameters()).device
        model = initialize_network(
            fabric=fabric, model_name=model_name, num_labels=num_labels, device=device
        )
        epochs = getattr(project_config, f"{dataset_name}_ViT_EPOCHS")
        milestones = getattr(project_config, f"{dataset_name}_ViT_MILESTONES")
    else:
        # ResNet18: recurse into all submodules to reset parameters.
        # `children()` only walks direct children (conv1, bn1, layer1, ...)
        # https://github.com/if-loops/selective-synaptic-dampening/blob/cdfdc0e35c1908e032a6e150d882b0fa17833f85/src/forget_random_strategies.py#L80
        # and would miss the Conv2d/BatchNorm2d nested inside each BasicBlock,
        # leaving most weights unchanged. `modules()` recurses through every
        # submodule so retrain truly starts from scratch.
        raw_model = model.module if hasattr(model, "module") else model
        for layer in raw_model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        epochs = getattr(project_config, f"{dataset_name}_RN_EPOCHS")
        milestones = getattr(project_config, f"{dataset_name}_RN_MILESTONES")

    # Capture the NEW wrapped model from fit_one_learning_cycle. Under FSDP,
    # fit_one_learning_cycle calls fabric.setup(raw_model, optimizer) internally
    # which creates a new FSDP wrapper; the original `model` parameter becomes
    # stale (its _forward_module references FlatParameters that got replaced
    # in-place). We must return the NEW wrapper so unlearn_main.py's
    # gather_full_state_dict() gets a valid state_dict.
    model, _ = fit_one_learning_cycle(
        fabric=fabric,
        num_gpus=num_gpus,
        epochs=epochs,
        model=model,
        train_dataloader=retain_train_dataloader,
        test_dataloader=retain_test_dataloader,
        milestones=milestones,
        lr=kwargs.get("lr", 0.1),
        wandb_logging_flag=wandb_logging_flag,
        neggrad_flag=False,
        model_name=model_name,
    )

    return model
