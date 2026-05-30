from supreme.utils.unlearning.unlearning_utils import fit_one_unlearning_cycle
import random
from lightning.fabric import Fabric
from supreme.utils.generic_utils import create_dataloader
import torch

"""
This method implements "Random Labels" as introduced in "Eternal Sunshine of the Spotless Net:
Selective Forgetting in Deep Networks" (https://arxiv.org/abs/1911.04933).

However, this function is also named 'amnesiac' in other codebases due to historical naming conventions incorrectly
"""


class RandomLabelDataset(torch.utils.data.Dataset):
    """Wraps forget + retain datasets, replacing forget labels with random ones on-the-fly
    instead of materializing all 50K samples into a Python list (~30GB for ViT)."""

    def __init__(self, forget_dataset, retain_dataset, random_labels):
        self.forget_dataset = forget_dataset
        self.retain_dataset = retain_dataset
        self.random_labels = random_labels
        self.n_forget = len(random_labels)

    def __len__(self):
        return self.n_forget + len(self.retain_dataset)

    def __getitem__(self, idx):
        if idx < self.n_forget:
            sample = self.forget_dataset[idx]
            return sample[0], sample[1], self.random_labels[idx]
        else:
            return self.retain_dataset[idx - self.n_forget]


def random_labeling(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    type_of_unlearning_strategy: str,
    num_labels: int,
    model: torch.nn.Module,  # Independent copy of the original model so it can go under the unlearning procedure
    retain_train_dataloader: torch.utils.data.DataLoader,
    retain_test_dataloader: torch.utils.data.DataLoader,
    forget_train_dataloader: torch.utils.data.DataLoader,
    **kwargs,
):
    unlearninglabels = list(range(num_labels))

    if (
        type_of_unlearning_strategy == "fullclass"
        or type_of_unlearning_strategy == "subclass"
    ):
        forget_label_id = kwargs.get("forget_label_id")
        if forget_label_id is not None:
            unlearninglabels.remove(forget_label_id)
        else:
            raise ValueError("forget_label_id is not provided")

    # Pre-generate random labels on rank 0 and broadcast to ensure GPU synchronization
    # This ensures all GPUs use the same random labels for the forget samples
    forget_dataset = forget_train_dataloader.dataset
    num_forget_samples = len(forget_dataset)

    if fabric.global_rank == 0:
        if type_of_unlearning_strategy == "random_":
            # For random_ strategy, pick labels that are NOT the sample's original class
            # (matching original logic: while rnd == clabel)
            random_labels = []
            for i in range(num_forget_samples):
                _, _, clabel = forget_dataset[i]
                rnd = random.choice(unlearninglabels)
                while rnd == clabel:
                    rnd = random.choice(unlearninglabels)
                random_labels.append(rnd)
        else:
            # For fullclass/subclass, pick any random label from available labels
            random_labels = [random.choice(unlearninglabels) for _ in range(num_forget_samples)]
        random_labels_tensor = torch.tensor(random_labels, device=fabric.device)
    else:
        random_labels_tensor = torch.zeros(num_forget_samples, dtype=torch.long, device=fabric.device)

    # Broadcast random labels from rank 0 to all ranks
    random_labels_tensor = fabric.broadcast(random_labels_tensor, src=0)
    random_labels = random_labels_tensor.cpu().tolist()

    # Use a Dataset wrapper instead of materializing all samples into a Python list
    unlearning_dataset = RandomLabelDataset(
        forget_dataset=forget_dataset,
        retain_dataset=retain_train_dataloader.dataset,
        random_labels=random_labels,
    )

    # Create the dataloader with appropriate settings
    unlearning_train_set_dataloader = create_dataloader(
        dataset=unlearning_dataset,
        batch_size=128,
        is_training=True,
        num_gpus=num_gpus,
    )

    unlearning_train_set_dataloader = fabric.setup_dataloaders(
        unlearning_train_set_dataloader
    )

    # Capture the NEW wrapped model (see retrain.py comment for FSDP rationale).
    model, _ = fit_one_unlearning_cycle(
        fabric=fabric,
        num_gpus=num_gpus,
        epochs=3,
        model=model,
        train_dataloader=unlearning_train_set_dataloader,
        test_dataloader=retain_test_dataloader,
        lr=0.0001,
        wandb_logging_flag=wandb_logging_flag,
    )

    return model
