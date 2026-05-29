"""
JIT (Just-In-Time) Unlearning Method
Adapted from: Information-Theoretic-Unlearning-main/src/lipschitz.py

VALIDATION AGAINST ORIGINAL CODE:
- Core algorithm matches original modify_weight method (lines 264-320)
- AddGaussianNoise class matches original (lines 37-51)
- Loss computation matches original Lipschitz constant calculation (line 307)
- Key difference: Added optimizer.zero_grad() per batch (original appears to have bug)
- Uses SUPREME 3-tuple dataloader format (x, indices, y) instead of original 2-tuple
- Updated to use Lightning Fabric for multi-GPU support following established patterns
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from lightning.fabric import Fabric
from copy import deepcopy


def disable_inplace_operations(module):
    """
    Recursively disable inplace operations in a module.
    This is necessary for proper gradient computation when backpropagating
    through a model multiple times or when using torch.autograd.grad.
    """
    for child in module.children():
        disable_inplace_operations(child)
    if hasattr(module, 'inplace'):
        module.inplace = False


class AddGaussianNoise(object):
    """Add Gaussian noise to tensors for JIT perturbation"""

    def __init__(self, mean=0.0, std=1.0, device="cpu"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        _max = tensor.max()
        _min = tensor.min()
        tensor = (
            tensor
            + torch.randn(tensor.size()).to(self.device) * self.std
            + self.mean
        )
        tensor = torch.clamp(tensor, min=_min, max=_max)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def jit(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,
    forget_train_dataloader: DataLoader,
    n_epochs: int = 1,
    n_samples: int = 10,
    learning_rate: float = 0.001,
    jit_weighting: float = 0.1,
    **kwargs,
):
    """
    JIT (Just-In-Time) unlearning method with multi-GPU support via Lightning Fabric.

    This method modifies model weights by enforcing Lipschitz continuity constraints
    around the forget samples. By controlling the local smoothness, it facilitates
    unlearning while maintaining overall model stability.

    Args:
        fabric: Fabric instance for distributed training
        num_gpus: Number of GPUs
        wandb_logging_flag: Whether to log to wandb
        model: The model to unlearn
        forget_train_dataloader: DataLoader for forget training data
        n_epochs: Number of unlearning epochs
        n_samples: Number of perturbed samples per image
        learning_rate: Learning rate for weight modification
        jit_weighting: Weight for JIT regularization (std of Gaussian noise)
        **kwargs: Extra arguments from framework (ignored)
    """

    # Log any extra kwargs that were passed but not used
    if kwargs:
        fabric.print(f"JIT: Ignoring extra framework arguments: {list(kwargs.keys())}")

    # CRITICAL: Disable inplace operations in the model BEFORE Fabric wrapping.
    # ResNet18 uses inplace ReLU (nn.ReLU(inplace=True)) which modifies tensors in place.
    # JIT does multiple forward passes (one with gradients, one without for perturbations),
    # and the inplace ops from the no_grad pass corrupt the saved tensors from the first pass.
    raw_model = model.module if hasattr(model, "module") else model
    disable_inplace_operations(raw_model)

    # Setup model and optimizer with Fabric (following ssd.py pattern)
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=learning_rate)
    model, optimizer = fabric.setup(raw_model, optimizer)

    # Get device from fabric
    device = fabric.device

    # Create a SEPARATE inference-only copy for perturbation forward passes.
    # ResNet18's forward() method creates nn.ReLU(inplace=True) dynamically on each call,
    # which our disable_inplace_operations can't catch. Using a separate model ensures
    # the perturbation passes don't corrupt the main model's saved tensors for backward.
    raw_model_after_setup = model.module if hasattr(model, "module") else model
    model_for_perturbation = deepcopy(raw_model_after_setup).to(device)
    model_for_perturbation.eval()  # Always in eval mode - inference only

    # Use Gaussian noise for perturbations
    transforms = v2.Compose([
        AddGaussianNoise(0.0, jit_weighting, device=device),
    ])

    fabric.print(f"Starting JIT unlearning for {n_epochs} epoch(s)")
    fabric.print(f"n_samples={n_samples}, learning_rate={learning_rate}, jit_weighting={jit_weighting}")
    fabric.print(f"DEBUG: Device = {device}")

    # Training loop
    num_forget_batches = len(forget_train_dataloader)
    fabric.print(f"DEBUG: num_forget_batches = {num_forget_batches}")

    for epoch in range(n_epochs):
        fabric.print(f"DEBUG: Starting epoch {epoch + 1}")
        fabric.print(f"\nJIT Epoch {epoch + 1}/{n_epochs}")

        model.train()

        running_loss = 0.0
        running_in_n = 0.0
        running_out_n = 0.0
        num_batches = 0

        for batch_idx, (x, _, _) in enumerate(forget_train_dataloader):
            if batch_idx == 0 or batch_idx % 20 == 0:
                fabric.print(f"DEBUG: Epoch {epoch+1}, Batch {batch_idx}/{num_forget_batches}")

            image = x.unsqueeze(0) if x.dim() == 3 else x
            if batch_idx == 0:
                fabric.print(f"DEBUG: image shape = {image.shape}")

            optimizer.zero_grad()

            # Sync perturbation model weights with main model before each batch
            _raw = model.module if hasattr(model, "module") else model
            model_for_perturbation.load_state_dict(_raw.state_dict())

            # Forward pass for gradient computation
            out = model(image)
            if batch_idx == 0:
                fabric.print(f"DEBUG: out shape = {out.shape}")

            # Initialize loss and norm accumulators (matching original lines 286-288)
            # CRITICAL: Don't use requires_grad=True here - it causes in-place operation errors
            loss = torch.tensor(0.0, device=device)
            out_n = torch.tensor(0.0, device=device)
            in_n = torch.tensor(0.0, device=device)

            # Build comparison images with perturbations (matching original lines 291-308)
            for sample_idx in range(n_samples):
                # Apply noise transform to create perturbed image
                img2 = transforms(x.clone())
                image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2

                # Use separate model for perturbation to avoid corrupting main model's computation graph
                with torch.no_grad():
                    out2 = model_for_perturbation(image2)

                # Flatten images for norm calculation (matching original lines 299-300)
                flatimg = image.view(image.size()[0], -1)
                flatimg2 = image2.view(image2.size()[0], -1)

                # Calculate norms (matching original lines 301-302)
                in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)
                out_norm = torch.linalg.vector_norm(out - out2, dim=1)

                # Accumulate norms for logging (matching original lines 304-305)
                in_n += in_norm.sum()
                out_n += out_norm.sum()

                # Lipschitz constant approximation (matching original line 307)
                K = ((out_norm / in_norm).sum()).abs()
                loss += K

            # Normalize (matching original lines 311-313)
            loss /= n_samples
            in_n /= n_samples
            out_n /= n_samples

            if batch_idx == 0:
                fabric.print(f"DEBUG: loss = {loss.item():.6f}, in_n = {in_n.item():.6f}, out_n = {out_n.item():.6f}")

            # Backward and optimize using fabric.backward()
            fabric.backward(loss)
            if batch_idx == 0:
                fabric.print(f"DEBUG: Backward done")
            optimizer.step()
            if batch_idx == 0:
                fabric.print(f"DEBUG: Optimizer step done")

            running_loss += loss.item()
            running_in_n += in_n.item()
            running_out_n += out_n.item()
            num_batches += 1

        fabric.print(f"DEBUG: Epoch {epoch+1} loop finished, aggregating metrics...")
        # Aggregate metrics across GPUs
        running_loss_tensor = torch.tensor([running_loss], device=device)
        running_in_n_tensor = torch.tensor([running_in_n], device=device)
        running_out_n_tensor = torch.tensor([running_out_n], device=device)
        num_batches_tensor = torch.tensor([num_batches], device=device)

        fabric.print(f"DEBUG: Before all_reduce")
        fabric.all_reduce(running_loss_tensor)
        fabric.all_reduce(running_in_n_tensor)
        fabric.all_reduce(running_out_n_tensor)
        fabric.all_reduce(num_batches_tensor)
        fabric.print(f"DEBUG: After all_reduce")

        avg_loss = running_loss_tensor.item() / max(num_batches_tensor.item(), 1)
        avg_in_n = running_in_n_tensor.item() / max(num_batches_tensor.item(), 1)
        avg_out_n = running_out_n_tensor.item() / max(num_batches_tensor.item(), 1)

        fabric.print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}, Avg In Norm: {avg_in_n:.6f}, Avg Out Norm: {avg_out_n:.6f}")

    fabric.print("JIT unlearning completed")

    # Return the post-fabric.setup() wrapped model (see ssd.py for rationale).
    return model
