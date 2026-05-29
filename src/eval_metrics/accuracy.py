import torch

"""
Paper: "Fast Yet Effective Machine Unlearning" at https://arxiv.org/abs/2111.08947
and implementation can be found at: https://github.com/vikram2000b/Fast-Machine-Unlearning/blob/main/Machine%20Unlearning.ipynb

Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" at https://arxiv.org/abs/2205.08096
and implementation can be found at: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/utils.py#L6

Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" at https://arxiv.org/abs/2308.07707
and implementation can be found at: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/utils.py#L9
"""


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100
