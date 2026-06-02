"""SCRUB (Selective Confusing Rote-learning Unlearning via Backtracking) unlearning method.

Notes:
Adapted from Information-Theoretic-Unlearning-main/supreme/scrub.py, and updated
to use Lightning Fabric for multi-GPU support following established patterns.

Validation against original code:
- Core algorithm matches original unlearning function (lines 116-209)
- DistillKL class matches original (lines 11-21), using modern reduction API
- Two-phase training: maximize on forget (lines 204-205), minimize on retain (line 206)
- sgda_adjust_learning_rate matches original (lines 35-43)
- Fixes applied to original bugs:
  * param_dist now compares student with teacher (original compared student with itself)
  * Optimizer creation fixed (original tried to call methods on string)
  * Uses SUPREME 3-tuple dataloader format (x, indices, y) instead of original 2-tuple

DDP compatibility notes:
- Teacher model has requires_grad=False to avoid storing activations (never trained)
- Unused model layers (e.g., ViT pooler) have requires_grad=False to prevent
  DDP "marked ready twice" errors from param_dist touching unused parameters
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from supreme.utils.fabric.fabric_setup import setup_model_for_inference


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network (matching original lines 11-21)"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # Original line 18-20, using modern reduction API instead of deprecated size_average
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        # Modern equivalent of: F.kl_div(..., size_average=False) * (T**2) / batch_size
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T**2)
        return loss


def sgda_adjust_learning_rate(
    epoch, sgda_learning_rate, lr_decay_epochs, lr_decay_rate, optimizer
):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    return new_lr


def disable_unused_parameters(model):
    """
    Disable gradients on model parameters that aren't used in the forward pass.

    This prevents DDP "marked ready twice" errors when param_dist touches parameters
    that don't participate in the normal forward-backward flow.

    Currently handles:
    - ViT pooler layer (HuggingFace ViT computes pooler but it's unused in classification)
    """
    disabled_count = 0
    for name, param in model.named_parameters():
        # ViT pooler is not used when taking last_hidden_state[:, 0] directly
        if "pooler" in name:
            param.requires_grad = False
            disabled_count += 1
    return disabled_count


def param_dist(model, model_ref, smoothing_weight):
    """
    Calculate parameter distance for smoothing regularization (matching original lines 45-50)

    NOTE: Original has a bug at line 106 where it calls param_dist(model_s, model_s, opt.smoothing)
    comparing the student with itself (distance always 0). This version correctly compares
    student with teacher as intended by the parameter names and smoothing logic.

    Args:
        model: Current model being trained (student)
        model_ref: Reference model (teacher, gradients will be detached)
        smoothing_weight: Weight for the smoothing term
    """
    dist = 0.0
    for p1, p2 in zip(model.parameters(), model_ref.parameters()):
        # Detach reference model parameters to avoid gradients flowing to it
        dist += torch.norm(p1 - p2.detach(), p="fro")
    return smoothing_weight * dist


def scrub(
    fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,
    retain_train_dataloader: DataLoader,
    forget_train_dataloader: DataLoader,
    sgda_epochs: int = 10,
    sgda_learning_rate: float = 0.0005,
    lr_decay_epochs: list = None,
    lr_decay_rate: float = 0.1,
    sgda_weight_decay: float = 0.1,
    sgda_momentum: float = 0.9,
    gamma: float = 1.0,
    alpha: float = 0.5,
    beta: float = 0.0,
    smoothing: float = 0.5,
    msteps: int = 3,
    kd_T: float = 2.0,
    optimizer_type: str = "adam",
    **kwargs,
):
    """
    SCRUB unlearning method with multi-GPU support via Lightning Fabric.

    Uses Smoothed Gradient Descent-Ascent (SGDA) to unlearn:
    - Gradient ascent on forget set (maximize divergence from teacher)
    - Gradient descent on retain set (maintain performance)

    Args:
        fabric: Fabric instance for distributed training
        num_gpus: Number of GPUs
        wandb_logging_flag: Whether to log to wandb
        model: The model to unlearn (will be modified in-place)
        retain_train_dataloader: DataLoader for retain training data
        forget_train_dataloader: DataLoader for forget training data
        sgda_epochs: Number of SGDA epochs
        sgda_learning_rate: Learning rate for SGDA
        lr_decay_epochs: Epochs at which to decay learning rate
        lr_decay_rate: Learning rate decay factor
        sgda_weight_decay: Weight decay factor
        sgda_momentum: Momentum for SGD optimizer
        gamma: Weight for classification loss
        alpha: Weight for KL divergence loss
        beta: Weight for additional KD loss
        smoothing: Smoothing parameter for parameter distance
        msteps: Number of maximize steps (gradient ascent) at the start
        kd_T: Temperature for knowledge distillation
        optimizer_type: Type of optimizer ("adam", "sgd", or "rmsp")
        **kwargs: Extra arguments from framework (ignored)
    """

    # Log any extra kwargs that were passed but not used
    if kwargs:
        fabric.print(
            f"SCRUB: Ignoring extra framework arguments: {list(kwargs.keys())}"
        )

    # Set lr_decay_epochs default
    if lr_decay_epochs is None:
        lr_decay_epochs = [5, 8, 9]

    # Create teacher model from original (inference-only)
    # MEMORY OPTIMIZATION: Set requires_grad=False to avoid storing activations
    raw_model = model.module if hasattr(model, "module") else model
    distributed_strategy_name = kwargs.get("distributed_strategy_name", "ddp")
    model_t = setup_model_for_inference(
        fabric, copy.deepcopy(raw_model), distributed_strategy_name
    )
    model_t.eval()
    for param in model_t.parameters():
        param.requires_grad = False

    # Setup criteria
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)

    # Disable unused parameters (e.g., ViT pooler) BEFORE creating optimizer
    # This prevents DDP "marked ready twice" errors from param_dist
    disabled = disable_unused_parameters(raw_model)
    if disabled > 0:
        fabric.print(
            f"Disabled gradients on {disabled} unused parameters (e.g., pooler)"
        )

    # Create optimizer for student model
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            raw_model.parameters(),
            lr=sgda_learning_rate,
            momentum=sgda_momentum,
            weight_decay=sgda_weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            raw_model.parameters(),
            lr=sgda_learning_rate,
            weight_decay=sgda_weight_decay,
        )
    elif optimizer_type == "rmsp":
        optimizer = torch.optim.RMSprop(
            raw_model.parameters(),
            lr=sgda_learning_rate,
            momentum=sgda_momentum,
            weight_decay=sgda_weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Setup model and optimizer with Fabric (following ssd.py pattern)
    model_s, optimizer = fabric.setup(raw_model, optimizer)  # type: ignore

    fabric.print(f"Starting SCRUB unlearning for {sgda_epochs} epochs")
    fabric.print(
        f"optimizer={optimizer_type}, lr={sgda_learning_rate}, msteps={msteps}"
    )
    fabric.print(f"gamma={gamma}, alpha={alpha}, smoothing={smoothing}")

    # Training loop
    for epoch in range(1, sgda_epochs + 1):
        lr = sgda_adjust_learning_rate(
            epoch, sgda_learning_rate, lr_decay_epochs, lr_decay_rate, optimizer
        )

        fabric.print(f"\nSCRUB Epoch {epoch}/{sgda_epochs} - LR: {lr:.6f}")

        # ============ MAXIMIZE PHASE ============
        # Maximize loss on forget set (gradient ascent) for first msteps epochs
        if epoch <= msteps:
            fabric.print("  Maximizing divergence on forget set...")
            model_s.train()

            running_loss = 0.0
            num_batches = 0

            for batch in forget_train_dataloader:
                input, _, target = batch

                # Forward pass
                logit_s = model_s(input)
                with torch.no_grad():
                    logit_t = model_t(input)

                # Calculate divergence loss
                loss_div = criterion_div(logit_s, logit_t)

                # Maximize divergence (minimize negative divergence)
                loss = -loss_div

                # Add smoothing regularization
                loss = loss + param_dist(model_s, model_t, smoothing)

                # Backward and optimize using fabric.backward()
                optimizer.zero_grad()
                fabric.backward(loss)
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            # Aggregate metrics across GPUs
            running_loss_tensor = torch.tensor([running_loss], device=fabric.device)
            num_batches_tensor = torch.tensor([num_batches], device=fabric.device)
            fabric.all_reduce(running_loss_tensor)
            fabric.all_reduce(num_batches_tensor)

            avg_maximize_loss = running_loss_tensor.item() / max(
                num_batches_tensor.item(), 1
            )
            fabric.print(f"  Maximize Loss: {avg_maximize_loss:.4f}")

        # ============ MINIMIZE PHASE ============
        # Minimize loss on retain set (gradient descent)
        fabric.print("  Minimizing loss on retain set...")
        model_s.train()

        running_loss = 0.0
        num_batches = 0

        for batch in retain_train_dataloader:
            input, _, target = batch

            # Forward pass
            logit_s = model_s(input)
            with torch.no_grad():
                logit_t = model_t(input)

            # Calculate losses
            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            # Retain set: minimize classification and divergence
            loss = gamma * loss_cls + alpha * loss_div

            # Add smoothing regularization
            loss = loss + param_dist(model_s, model_t, smoothing)

            # Backward and optimize using fabric.backward()
            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        # Aggregate metrics across GPUs
        running_loss_tensor = torch.tensor([running_loss], device=fabric.device)
        num_batches_tensor = torch.tensor([num_batches], device=fabric.device)
        fabric.all_reduce(running_loss_tensor)
        fabric.all_reduce(num_batches_tensor)

        avg_minimize_loss = running_loss_tensor.item() / max(
            num_batches_tensor.item(), 1
        )
        fabric.print(f"  Minimize Loss: {avg_minimize_loss:.4f}")

    fabric.print("SCRUB unlearning completed")

    # Return the trained student model (see ssd.py for FSDP rationale).
    return model_s
