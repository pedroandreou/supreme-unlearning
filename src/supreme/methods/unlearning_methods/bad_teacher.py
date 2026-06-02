"""Bad Teacher (BadT) unlearning method.

Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" (https://arxiv.org/abs/2205.08096)
Reference: https://github.com/vikram2000b/bad-teaching-unlearning/
Reference: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/unlearn.py#L9
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/unlearn.py#L11
"""

import torch
import random
from supreme.datasets.datasets import CombinedForgetRetainDataset
from torch.nn import functional as F
import torch.nn as nn
import wandb
import numpy as np
from lightning.fabric import Fabric
from supreme.utils.generic_utils import create_dataloader
from copy import deepcopy
from torch.utils.data import DataLoader
from supreme.utils.fabric.fabric_setup import setup_model_for_inference


class UnlearnerLossClass(nn.Module):
    def __init__(self, KL_temperature):
        super().__init__()
        self.KL_temperature = KL_temperature

    def forward(self, output, labels, full_teacher_logits, unlearn_teacher_logits):
        labels = torch.unsqueeze(labels, dim=1)
        f_teacher_out = F.softmax(full_teacher_logits / self.KL_temperature, dim=1)
        u_teacher_out = F.softmax(unlearn_teacher_logits / self.KL_temperature, dim=1)
        # label 1 means forget sample
        # label 0 means retain sample
        overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
        student_out = F.log_softmax(output / self.KL_temperature, dim=1)
        return F.kl_div(student_out, overall_teacher_out, reduction="mean")


# https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/unlearn.py#L21
# and
# https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/unlearn.py#L26
def unlearning_step(
    fabric,
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_dataloader,
    optimizer,
    criterion,
    epoch,
):
    # if fabric.global_rank == 0:
    #     fabric.call("on_train_epoch_start", fabric=fabric, epoch=epoch)
    # fabric.barrier()

    losses = []
    for batch_idx, batch in enumerate(unlearn_dataloader):
        # if fabric.global_rank == 0:
        #     fabric.call("on_train_batch_start")
        # fabric.barrier()

        x, y = batch

        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)

        output = model(x)
        optimizer.zero_grad()

        loss = criterion(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
        )

        fabric.backward(loss)
        optimizer.step()

        # current_lr = optimizer.param_groups[0]["lr"]
        losses.append(loss.detach().cpu().numpy())

        # if fabric.global_rank == 0:
        #     fabric.call(
        #         "on_train_batch_end",
        #         loss=loss,
        #         epoch=epoch,
        #         batch_idx=batch_idx,
        #         lr=current_lr,
        #     )
        # fabric.barrier()

    avg_loss = np.mean(losses)  # Simple mean is fine here since it's training

    # if fabric.global_rank == 0:
    #     # Call epoch end
    #     fabric.call(
    #         "on_train_epoch_end",
    #         epoch=epoch,
    #         train_loss=avg_loss,
    #         last_lr=current_lr,
    #     )
    # fabric.barrier()

    return avg_loss


# https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/unlearn.py#L67
# and
# https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/unlearn.py#L84
def bad_teacher_unlearner(
    fabric,
    num_gpus,
    model,  # student_model
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    epochs=10,
    optimizer="adam",
    lr=0.01,
    batch_size=256,
    KL_temperature=1,
    wandb_logging_flag=False,
):
    # Creating the unlearning dataset
    unlearning_data = CombinedForgetRetainDataset(
        forget_data=forget_data, retain_data=retain_data
    )

    # Create the dataloader with appropriate settings
    unlearning_dataloader = create_dataloader(
        dataset=unlearning_data,
        batch_size=batch_size,
        is_training=True,
        num_gpus=num_gpus,
    )
    unlearning_dataloader = fabric.setup_dataloaders(unlearning_dataloader)

    # Create and setup criterion
    criterion = UnlearnerLossClass(KL_temperature=KL_temperature)

    if fabric.global_rank == 0 and wandb_logging_flag:
        # Update the config
        config_dict = {
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "learning_rate": lr,
            "KL_temperature": KL_temperature,
            "batch_size": batch_size,
        }
        wandb.config.update(config_dict)
    fabric.barrier()

    model.train()
    unlearning_teacher.eval()
    full_trained_teacher.eval()

    for epoch in range(epochs):
        loss = unlearning_step(
            fabric=fabric,
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_dataloader=unlearning_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        fabric.print("Epoch {} Unlearning Loss {}".format(epoch + 1, loss))


# https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/forget_random_strategies.py#L137
def bad_teacher(  # in the bad teacher repo this method is named as blindspot but there is a regression unlearning method that is named like that so we renamed it to bad teacher
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,  # Independent copy of the original model so it can go under the unlearning procedure
    unlearning_teacher: nn.Module,
    retain_train_dataloader: DataLoader,
    forget_train_dataloader: DataLoader,
    **kwargs,
):
    lr = 0.0001
    raw_model = model.module if hasattr(model, "module") else model
    distributed_strategy_name = kwargs.get("distributed_strategy_name", "ddp")
    # deepcopy BEFORE fabric.setup - after FSDP wrapping, submodules contain NCCL
    # process groups which cannot be pickled (TypeError: cannot pickle 'module' object).
    teacher_copy = deepcopy(raw_model)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr)
    model, optimizer = fabric.setup(raw_model, optimizer)

    full_trained_teacher = setup_model_for_inference(
        fabric, teacher_copy, distributed_strategy_name
    )

    b_s = 256

    KL_temperature = 1

    # Synchronize random sampling across all GPUs to ensure consistent training data
    # Only rank 0 generates indices, then broadcasts to all other ranks
    retain_dataset = retain_train_dataloader.dataset
    dataset_len = len(retain_dataset)
    subset_size = int(0.3 * dataset_len)

    if fabric.global_rank == 0:
        # Generate random indices on rank 0
        indices = random.sample(range(dataset_len), subset_size)
        indices_tensor = torch.tensor(indices, device=fabric.device)
    else:
        # Create placeholder tensor on other ranks
        indices_tensor = torch.zeros(
            subset_size, dtype=torch.long, device=fabric.device
        )

    # Broadcast indices from rank 0 to all other ranks
    indices_tensor = fabric.broadcast(indices_tensor, src=0)
    indices = indices_tensor.cpu().tolist()

    # Use Subset instead of materializing all samples into a Python list
    retain_train_subset = torch.utils.data.Subset(retain_dataset, indices)

    bad_teacher_unlearner(
        fabric=fabric,
        num_gpus=num_gpus,
        model=model,  # it is the original model that will go under the unlearning procedure, it is the same with the fully trained model
        unlearning_teacher=unlearning_teacher,  # the incompetent teacher is a randomly initialized model
        full_trained_teacher=full_trained_teacher,  # the competent teacher is the fully trained model or the original model
        retain_data=retain_train_subset,
        forget_data=forget_train_dataloader.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=lr,
        batch_size=b_s,
        KL_temperature=KL_temperature,
        wandb_logging_flag=wandb_logging_flag,
    )

    # Return the post-fabric.setup() wrapped model (see ssd.py for rationale).
    return model
