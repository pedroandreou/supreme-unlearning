"""SUPREME - a registry-based, multi-GPU framework for reproducible
image-unlearning evaluation.

Public API
==========

Registration (extend the framework from your own package, no edits to SUPREME)::

    import supreme
    supreme.register_model("MyNet", "my_pkg.models:MyNet")
    supreme.register_baseline("mybase", "my_pkg.mybase")
    supreme.register_unlearning_method("mymethod", "my_pkg.mymethod")
    supreme.register_metric("mymetric", "my_pkg.mymetric", requires_retrain=False)
    supreme.register_dataset("MyDS", "my_pkg.data:MyDS",
                             root="/data/myds", class_dict={"a": 0, "b": 1})

An installed plugin package can equivalently provide components via packaging
entry points - see ``supreme.registry`` and ``docs/extending.md``.

Running the pipeline::

    supreme.run_training(["-net", "ViT", "-dataset", "Cifar10", "-seed", "260"])
    supreme.run_unlearning(["-method", "ssd", "-net", "ViT", "-dataset", "Cifar10"])

or use the installed console scripts ``supreme-train`` / ``supreme-unlearn``.

Configuration & registries are exposed via ``supreme.project_config``.

This module imports only the registry and the (torch-free) ``project_config``,
so registration is available without importing the heavy PyTorch/Lightning
stack. The pipeline entry points (``run_training`` / ``run_unlearning``) import
that stack lazily on first call.
"""

from __future__ import annotations

import sys

from supreme.registry import (
    PACKAGE_ROOT,
    register_baseline,
    register_dataset,
    register_metric,
    register_model,
    register_unlearning_method,
)
from supreme.utils import project_config

__version__ = "0.1.3"

__all__ = [
    "__version__",
    "PACKAGE_ROOT",
    "project_config",
    "register_model",
    "register_baseline",
    "register_unlearning_method",
    "register_metric",
    "register_dataset",
    "run_training",
    "run_unlearning",
]


def _run_with_argv(main_func, argv, prog):
    """Invoke an argparse ``main()`` with an explicit argv list.

    The training/unlearning entry points parse ``sys.argv`` via argparse. This
    helper lets callers pass arguments programmatically without spawning a
    subprocess, while leaving ``sys.argv`` untouched afterwards.
    """
    if argv is None:
        return main_func()
    saved = sys.argv
    sys.argv = [prog, *list(argv)]
    try:
        return main_func()
    finally:
        sys.argv = saved


def run_training(argv=None):
    """Run the training stage. ``argv``: list of CLI args (see ``supreme-train -h``)."""
    from supreme.utils.training.train_main import main

    return _run_with_argv(main, argv, "supreme-train")


def run_unlearning(argv=None):
    """Run the unlearn/evaluate stage. ``argv``: list of CLI args (see ``supreme-unlearn -h``)."""
    from supreme.utils.unlearning.unlearn_main import main

    return _run_with_argv(main, argv, "supreme-unlearn")
