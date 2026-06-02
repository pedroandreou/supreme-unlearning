"""Eval-metric math tests with known closed-form answers.

These exercise the undecorated, pure-tensor metrics on CPU. They assert
mathematical ground truth (perfect accuracy is 100, the JS divergence of a
distribution with itself is 0) rather than reproducing the implementation, so
they catch sign flips, wrong reductions, and axis mistakes.
"""

import torch

from supreme.eval_metrics.accuracy import accuracy
from supreme.eval_metrics.jsdiv import JSDiv


def test_accuracy_all_correct_is_100():
    logits = torch.eye(4)  # argmax of row i is i
    assert accuracy(logits, torch.arange(4)).item() == 100.0


def test_accuracy_all_wrong_is_0():
    logits = torch.eye(2)
    assert accuracy(logits, torch.tensor([1, 0])).item() == 0.0


def test_accuracy_half_is_50():
    logits = torch.eye(4)
    labels = torch.tensor([0, 1, 0, 0])  # 2 of 4 correct
    assert accuracy(logits, labels).item() == 50.0


def test_jsdiv_of_identical_distributions_is_zero():
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]])
    assert JSDiv(None, p, p, do_global_aggregation=False) == 0.0


def test_jsdiv_is_nonnegative_for_differing_distributions():
    p = torch.tensor([[0.7, 0.1, 0.1, 0.1]])
    q = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
    assert JSDiv(None, p, q, do_global_aggregation=False) > 0.0


def test_jsdiv_global_aggregation_returns_dict(fake_fabric):
    p = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    out = JSDiv(fake_fabric, p, p, do_global_aggregation=True)
    assert out["final_value"] == 0.0
    assert "per_process" in out
