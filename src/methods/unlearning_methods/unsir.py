import torch
from torch.nn import functional as F
from src.utils.unlearning.unlearning_utils import (
    fit_one_unlearning_cycle,
    get_classwise_indices,
)
import wandb
from lightning.fabric import Fabric
from src.utils.generic_utils import create_dataloader


# Paper: "Fast Yet Effective Machine Unlearning" at https://arxiv.org/abs/2111.08947
# and implementation can be found at https://github.com/vikram2000b/Fast-Machine-Unlearning/blob/main/Machine%20Unlearning.ipynb
class UNSIR_noise(torch.nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self, _dummy_input: torch.Tensor = None):
        # Fabric/DDP compatibility: accept an optional dummy tensor so the wrapped
        # module always receives a non-empty input tuple. The value is ignored; we
        # simply return the learnable noise tensor.
        # Important: return a non-leaf view/clone to avoid DDP marking the same
        # parameter "ready" multiple times when it's used along multiple paths
        # in the loss (e.g., via the model forward and an L2 regularizer).
        return self.noise.clone()


def UNSIR_noise_train(
    fabric, num_gpus, noise, model, forget_label_id, num_epochs, noise_batch_size
):
    """
    # the explanation can be found at:
    https://github.com/vikram2000b/bad-teaching-unlearning/blob/6fe4b032761314ba3874bdb6d651583a45e15f55/unlearn.py#L105

    The UNSIR_noise_train function trains the noise tensor to maximize a specific loss function.
    All processes train in parallel and we synchronize the noise parameters at the end
    to ensure identical noise across processes.
    """
    model.eval()  # We're not training the model, just using it to generate gradients

    opt = torch.optim.Adam(noise.parameters(), lr=0.1)
    # # Move noise to the correct device if not already there
    # noise = noise.to(fabric.device)

    noise, opt = fabric.setup(noise, opt)

    for epoch in range(num_epochs):
        total_loss = []
        # Fabric/DDP compatibility: pass a dummy tensor so wrappers get a non-empty
        # input tuple and don't crash on empty args.
        dummy = torch.empty(1, device=fabric.device)
        inputs = noise(dummy)
        # Create labels directly on the correct device
        labels = torch.zeros(noise_batch_size, device=fabric.device) + forget_label_id

        outputs = model(inputs)
        loss = -F.cross_entropy(
            outputs, labels.long(), reduction="mean"
        ) + 0.1 * torch.mean(torch.sum(inputs**2, [1, 2, 3]))

        opt.zero_grad()
        fabric.backward(loss)
        opt.step()

        total_loss.append(loss.cpu().detach().numpy())
        if fabric.global_rank == 0 and epoch % 5 == 0:
            fabric.print(f"Loss: {loss.item()}")
        fabric.barrier()

    return noise


class NoisyRetainDataset(torch.utils.data.Dataset):
    """Combines generated noise samples with retain samples for the UNSIR impair step.
    The noise portion is stored as a contiguous tensor to avoid per-sample Python
    object overhead."""

    def __init__(self, noise_tensor, forget_label_id, retain_dataset):
        self.noise_tensor = noise_tensor  # (N, C, H, W) pre-generated on CPU
        self.forget_label_id = forget_label_id
        self.retain_dataset = retain_dataset
        self.n_noise = noise_tensor.size(0)

    def __len__(self):
        return self.n_noise + len(self.retain_dataset)

    def __getitem__(self, idx):
        if idx < self.n_noise:
            return self.noise_tensor[idx], self.forget_label_id, self.forget_label_id
        else:
            sample = self.retain_dataset[idx - self.n_noise]
            return sample[0], sample[2], sample[2]


class RetainRelabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset returning (image, clabel, clabel) for the UNSIR repair step,
    without copying samples into a new list."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return sample[0], sample[2], sample[2]


def UNSIR_create_noisy_dataloader(
    num_gpus: int,
    noise,
    forget_label_id,
    retain_dataset,
    batch_size,
    num_noise_batches=80,
):
    """
    the explanation can be found at:
    https://github.com/vikram2000b/bad-teaching-unlearning/blob/6fe4b032761314ba3874bdb6d651583a45e15f55/unlearn.py#L151
    """
    # Generate all noise as a single contiguous tensor instead of a list of tuples
    dummy = torch.empty(1, device=noise.noise.device)
    noise_batches = []
    for _ in range(num_noise_batches):
        batch = noise(dummy)
        noise_batches.append(batch.detach().cpu())
    noise_tensor = torch.cat(noise_batches, dim=0)

    dataset = NoisyRetainDataset(noise_tensor, forget_label_id, retain_dataset)

    # Create the dataloader with appropriate settings
    noisy_dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        is_training=True,
        num_gpus=num_gpus,
    )

    return noisy_dataloader


##########################################################################
########### THIS METHOD IS USED ONLY IN FULLCLASS AND SUBCLASS ###########
############################## NOT IN RANDOM #############################
##########################################################################


def unsir(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    type_of_unlearning_strategy: str,
    num_labels: int,
    forget_label_id: int,
    model: torch.nn.Module,  # Independent copy of the original model so it can go under the unlearning procedure
    trainset: torch.utils.data.Dataset,
    retain_train_dataloader: torch.utils.data.DataLoader,
    retain_test_dataloader: torch.utils.data.DataLoader,
    **kwargs,
):
    num_noise_iterations = 25 if type_of_unlearning_strategy == "fullclass" else 250
    num_samples = 500
    noise_batch_size = 32
    num_noise_batches = 80

    if fabric.global_rank == 0 and wandb_logging_flag:
        # Update the config with UNSIR-specific parameters
        config_dict = {
            "unlearning_strategy": type_of_unlearning_strategy,
            "forget_label_id": forget_label_id,
            "num_samples": num_samples,
            "noise_batch_size": noise_batch_size,
            "num_noise_iterations": num_noise_iterations,
            "num_noise_batches": num_noise_batches,
        }
        wandb.config.update(config_dict)
    fabric.barrier()

    classwise_train_indices = get_classwise_indices(
        trainset, num_labels, type_of_unlearning_strategy
    )

    # Use Subset instead of materializing all samples into a Python list.
    # This is safe because trainset is created with unlearning=True (deterministic
    # transforms: ToTensor + Normalize only, no random augmentation), so repeated
    # calls to __getitem__ on the same index produce identical tensors.
    retain_indices = []
    for i in range(num_labels):
        if i != forget_label_id:
            retain_indices.extend(classwise_train_indices[i][:num_samples])

    retain_samples = torch.utils.data.Subset(trainset, retain_indices)
    fabric.print(f"Length of retain_samples: {len(retain_samples)}")

    img_shape = next(iter(retain_train_dataloader.dataset))[0].shape[-1]

    # Create noise on rank 0 only and broadcast to ensure identical initial state across GPUs
    noise = None
    if fabric.global_rank == 0:
        noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape)
    fabric.barrier()
    noise = fabric.broadcast(noise, src=0)
    fabric.print("Synchronized initial UNSIR noise across all GPUs")

    # Train noise
    noise = UNSIR_noise_train(
        fabric,
        num_gpus,
        noise,
        model,
        forget_label_id,
        num_noise_iterations,
        noise_batch_size,
    )
    noisy_dataloader = UNSIR_create_noisy_dataloader(
        num_gpus,
        noise,
        forget_label_id,
        retain_samples,
        noise_batch_size,
        num_noise_batches,
    )

    retain_test_dataloader = create_dataloader(
        dataset=retain_test_dataloader.dataset,
        batch_size=noise_batch_size,
        is_training=False,
        num_gpus=num_gpus,
    )
    noisy_dataloader, retain_test_dataloader = fabric.setup_dataloaders(
        noisy_dataloader,
        retain_test_dataloader,
    )

    # Impair Step. Capture the NEW wrapped model - see retrain.py for rationale.
    fabric.print("Starting Impair Step...")
    model, _ = fit_one_unlearning_cycle(
        fabric=fabric,
        num_gpus=num_gpus,
        epochs=1,
        model=model,
        train_dataloader=noisy_dataloader,
        test_dataloader=retain_test_dataloader,
        lr=0.0001,
        wandb_logging_flag=wandb_logging_flag,
    )

    # Repair Step
    fabric.print("Starting Repair Step...")
    heal_dataset = RetainRelabelDataset(retain_samples)

    heal_dataloader = create_dataloader(
        dataset=heal_dataset,
        batch_size=128,
        is_training=True,
        num_gpus=num_gpus,
    )
    heal_dataloader = fabric.setup_dataloaders(heal_dataloader)

    model, _ = fit_one_unlearning_cycle(
        fabric=fabric,
        num_gpus=num_gpus,
        epochs=1,
        model=model,
        train_dataloader=heal_dataloader,
        test_dataloader=retain_test_dataloader,
        lr=0.0001,
        wandb_logging_flag=wandb_logging_flag,
    )

    return model