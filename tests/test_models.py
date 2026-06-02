"""Model construction + forward-shape smoke tests (CPU, tiny tensors).

These prove a model builds and its forward pass honours the
``(batch, num_labels)`` output contract every metric and loss downstream
assumes. ViT is intentionally not covered: its backbone is fetched from the
Hugging Face Hub on construction, which we keep out of CI. ResNet18 is fully
local and CPU-friendly.
"""

import torch

from supreme.models.ResNet18 import ResNet18


def test_resnet18_forward_output_shape():
    model = ResNet18(num_labels=10).eval()
    images = torch.randn(2, 3, 32, 32)  # CIFAR-sized batch of 2
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (2, 10)


def test_resnet18_respects_num_labels():
    model = ResNet18(num_labels=37).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 37)
