"""Optimizer-free replicated inference for DeepSpeed ZeRO 1/2 selections."""

import torch
from lightning_utilities import apply_to_collection


class ReplicatedEvaluationModule(torch.nn.Module):
    """Match DeepSpeed's forward precision without allocating training state.

    DeepSpeed 0.15 creates an unused BF16 optimizer/gradient buffer even with
    optimizer=None and ZeRO disabled. ZeRO 1/2 have no state to shard during
    evaluation, so use ordinary replicated forwards within the existing
    DeepSpeed process group. Dataloaders and metric collectives still use Fabric.
    """

    def __init__(self, fabric, module):
        super().__init__()
        precision = fabric.strategy.precision.precision
        self.compute_dtype = {
            "32-true": torch.float32,
            "16-true": torch.float16,
            "16-mixed": torch.float16,
            "bf16-true": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
        }[precision]
        self.module = module.to(device=fabric.device, dtype=self.compute_dtype).eval()
        self.module.requires_grad_(False)
        # Synchronize both checkpoint parameters and inference buffers once.
        # Do not synchronize every forward; samples are independent in eval mode.
        with torch.no_grad():
            for value in (*self.module.parameters(), *self.module.buffers()):
                value.copy_(fabric.broadcast(value, src=0))
        self.eval()

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        def convert(value):
            return value.to(self.compute_dtype) if value.is_floating_point() else value

        args, kwargs = apply_to_collection((args, kwargs), torch.Tensor, convert)
        output = self.module(*args, **kwargs)
        return apply_to_collection(
            output,
            torch.Tensor,
            lambda value: value.to(torch.get_default_dtype())
            if value.is_floating_point()
            else value,
        )
