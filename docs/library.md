# SUPREME as a Python library

SUPREME is a **pip-installable Python library** (`import supreme`), not just a
set of scripts. Install it, register your own components, and drive the full
**train → unlearn → evaluate** pipeline from Python, with no edits to the
framework:

```bash
pip install supreme-unlearning
```

```python
import supreme

# Run the built-in pipeline programmatically
supreme.run_training(["-net", "ViT", "-dataset", "Cifar10", "-seed", "260"])
supreme.run_unlearning(["-method", "ssd", "-net", "ViT", "-dataset", "Cifar10"])

# Plug in code you wrote yourself, living in your own package.
# Replace "your_package.your_method" with your real import path.
supreme.register_unlearning_method("mymethod", "your_package.your_method")
supreme.run_unlearning(["-method", "mymethod", "-net", "ViT", "-dataset", "Cifar10"])
```

**Public API:** `supreme.run_training`, `supreme.run_unlearning`,
`supreme.register_model`, `supreme.register_baseline`,
`supreme.register_unlearning_method`, `supreme.register_metric`,
`supreme.register_dataset`, and `supreme.project_config`. Everything under
`supreme.utils.*` is internal. The API is defined in
[`src/supreme/__init__.py`](../src/supreme/__init__.py); resolution and plugin entry points
live in [`src/supreme/registry.py`](../src/supreme/registry.py). Full walkthrough:
[`docs/extending.md`](extending.md) and the notebook
[`notebooks/custom_components.ipynb`](../notebooks/custom_components.ipynb).

## Where the code lives

| Path | What's there |
|---|---|
| [`src/supreme/__init__.py`](../src/supreme/__init__.py) | Public API surface (`run_*`, `register_*`) |
| [`src/supreme/registry.py`](../src/supreme/registry.py) | Name → component resolution and plugin entry points |
| [`src/supreme/methods/unlearning_methods/`](../src/supreme/methods/unlearning_methods/) | Unlearning method implementations |
| [`src/supreme/methods/baselines/`](../src/supreme/methods/baselines/) | Retrain baseline + Original (no-unlearning) reference |
| [`src/supreme/models/`](../src/supreme/models/) | ResNet18, ViT |
| [`src/supreme/datasets/datasets.py`](../src/supreme/datasets/datasets.py) | The 5 datasets |
| [`src/supreme/eval_metrics/`](../src/supreme/eval_metrics/) | Evaluation metric implementations |
| [`src/supreme/utils/training/train_main.py`](../src/supreme/utils/training/train_main.py) | Training-stage entry point (`supreme-train`) |
| [`src/supreme/utils/unlearning/unlearn_main.py`](../src/supreme/utils/unlearning/unlearn_main.py) | Unlearn/evaluate entry point (`supreme-unlearn`) |
| [`src/supreme/utils/fabric/`](../src/supreme/utils/fabric/) | Lightning Fabric setup (accelerators, precision, distributed strategies) |
