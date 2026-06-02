<div align="center">

<h3><strong>⚡ SUPREME - A Multi-GPU Framework for Reproducible Image Unlearning Method Evaluation</strong></h3>

![*SUPREME*](assets/SUPREME-wordmark.svg)

<p>
  <strong>🔬 Tech Stack</strong><br>
  <em>Core:</em>
  <a href="#"><img src="https://img.shields.io/badge/python-3.9-blue.svg?logo=python&logoColor=white" alt="Python 3.9"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://lightning.ai/docs/fabric/"><img src="https://img.shields.io/badge/Lightning_Fabric-792EE5?logo=lightning&logoColor=white" alt="Lightning Fabric"></a>
  <a href="https://huggingface.co/docs/transformers/"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace Transformers"></a>
  <br>
  <em>Accelerators:</em>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA_12.1-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.1"></a>
  <a href="https://developer.apple.com/metal/pytorch/"><img src="https://img.shields.io/badge/MPS-000000?logo=apple&logoColor=white" alt="MPS"></a>
  <a href="https://pytorch.org/xla/"><img src="https://img.shields.io/badge/TPU_(XLA)-4285F4?logo=googlecloud&logoColor=white" alt="TPU via PyTorch XLA"></a>
  <br>
  <em>Distributed & precision:</em>
  <a href="https://www.deepspeed.ai/"><img src="https://img.shields.io/badge/DeepSpeed-0078D4?logo=microsoft&logoColor=white" alt="DeepSpeed"></a>
  <a href="https://huggingface.co/docs/bitsandbytes/"><img src="https://img.shields.io/badge/bitsandbytes-FFD21E?logo=huggingface&logoColor=black" alt="bitsandbytes"></a>
  <a href="https://github.com/NVIDIA/TransformerEngine"><img src="https://img.shields.io/badge/TransformerEngine-76B900?logo=nvidia&logoColor=white" alt="NVIDIA TransformerEngine"></a>
</p>

<p>
  <strong>🛠️ Tooling</strong><br>
  <em>Experiment tracking:</em>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Weights_%26_Biases-FFBE00?logo=weightsandbiases&logoColor=black" alt="Weights & Biases"></a>
  <a href="https://www.tensorflow.org/tensorboard"><img src="https://img.shields.io/badge/TensorBoard-FF6F00?logo=tensorflow&logoColor=white" alt="TensorBoard"></a>
  <br>
  <em>Environment:</em>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker"></a>
  <a href="https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/pedroandreou/supreme-unlearning"><img src="https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode" alt="Open in Dev Containers"></a>
  <br>
  <em>Debug & profile:</em>
  <a href="https://github.com/microsoft/debugpy"><img src="https://img.shields.io/badge/debugpy-007ACC?logo=visualstudiocode&logoColor=white" alt="debugpy"></a>
  <a href="https://github.com/plasma-umass/scalene"><img src="https://img.shields.io/badge/Scalene_Profiler-6A0DAD?logo=python&logoColor=white" alt="Scalene Profiler"></a>
  <br>
  <em>Code quality:</em>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/Ruff-D7FF64?logo=ruff&logoColor=black" alt="Ruff"></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-FAB040?logo=precommit&logoColor=white" alt="pre-commit"></a>
</p>

<p>
  <strong>📄 Publication</strong><br>
  <a href="https://arxiv.org/abs/2606.00380"><img src="https://img.shields.io/badge/arXiv-2606.00380-b31b1b?logo=arxiv&logoColor=white" alt="arXiv Preprint"></a>
  <a href="https://aiimlab.org/events/ECML_PKDD_2026_WIPE-OUT_2_Workshop_on_Machine_Unlearning_and_Privacy_Preservation.html"><img src="https://img.shields.io/badge/Under_Review-WIPE--OUT_2_(ECML--PKDD_2026)-yellow" alt="Under double-blind review at the WIPE-OUT 2 Workshop, ECML-PKDD 2026"></a>
  <a href="https://pedroandreou.github.io/supreme-unlearning-page/"><img src="https://img.shields.io/badge/Project_Page-Live-2ea44f?logo=githubpages&logoColor=white" alt="Project Page"></a>
</p>

<p>
  <strong>📦 Repository</strong><br>
  <a href="https://github.com/pedroandreou/supreme-unlearning/actions/workflows/ci.yml"><img src="https://github.com/pedroandreou/supreme-unlearning/actions/workflows/ci.yml/badge.svg" alt="CI (lint, build, tests)"></a>
  <a href="https://pypi.org/project/supreme-unlearning/"><img src="https://img.shields.io/pypi/v/supreme-unlearning?logo=pypi&logoColor=white&label=PyPI&cacheSeconds=3600" alt="PyPI"></a>
  <a href="https://test.pypi.org/project/supreme-unlearning/"><img src="https://img.shields.io/badge/TestPyPI-supreme--unlearning-orange?logo=pypi&logoColor=white" alt="TestPyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT License"></a>
</p>

</div>

---

## 📖 Overview

**SUPREME** is an open-source framework for evaluating *machine unlearning* methods on image classification tasks at scale.

Machine unlearning removes the influence of a chosen subset of training data (a class, a sub-class, or a random sample) from an already-trained model, *without* retraining from scratch. A good unlearned model should behave as if it had never seen the forgotten data while still classifying everything else accurately. Comparing the many proposed methods fairly demands a standardised, repeatable harness, and SUPREME is that harness.

**The gap it fills.** Existing image-classification unlearning frameworks - [MUBox](https://dl.acm.org/doi/10.1145/3734436.3734454), [DeepUnlearn](https://github.com/xcadet/deepunlearn), and [ERASURE](https://github.com/aiim-research/ERASURE) - run on a single device, which caps how many methods, scenarios, and seeds can be evaluated in reasonable time. SUPREME distributes the **entire** train → unlearn → evaluate pipeline across multiple GPUs and nodes, removing that bottleneck. It does for image-classification unlearning what [Open-Unlearning](https://github.com/locuslab/open-unlearning) did for LLM unlearning in the text domain: turn a single-device research problem into a scalable, reproducible benchmark. To our knowledge it is the first multi-GPU framework for the field.

**What it offers out of the box:**

- **A complete, automated pipeline.** Train a baseline on the full dataset, unlearn the chosen subset with the selected method, then evaluate the result against a from-scratch *retrained* reference, all from one command. Re-runs detect and skip work that is already done.
- **A broad component library.** **5 datasets, 2 model architectures, 2 baselines, 9 unlearning methods, 9 evaluation metrics** (covering forgetting, utility, privacy, behavioural/parametric equivalence, and efficiency), and **3 unlearning scenarios** (full-class, subclass, random-sample), all selectable through command-line flags.
- **Distributed, multi-precision execution.** Built on PyTorch and Lightning Fabric. DDP, FSDP, and DeepSpeed ZeRO 1/2/3 apply to *all three stages*, with mixed precision (fp16 / bf16, FP8, 4-/8-bit) and CUDA / Apple Silicon (MPS) / TPU / CPU back-ends. SLURM helpers fan experiments out across a cluster.
- **Statistically honest evaluation.** A single random seed [misrepresents how an unlearning method really behaves](https://arxiv.org/abs/2510.26714), because randomness enters at three independent points: **training** (weight initialisation and data shuffling produce different base models), **unlearning** (the unlearning algorithm itself is stochastic), and **evaluation** (sampling and metric computation add their own noise). SUPREME varies the seed at each of these three stages separately, so you can see how much of the spread in a result comes from the base model, from the unlearning run, and from measurement, and report the full distribution rather than a single point estimate. The seed count at each stage is configurable per run.
- **Extensibility without forking.** It is pip-installable (`pip install supreme-unlearning`) and registry-based: add a dataset, model, method, or metric from your own package by implementing a small interface and registering its module path, with no edits to framework code (see [`docs/extending.md`](docs/extending.md)).
- **Efficient reuse.** Experiments that share a training configuration train the model once and reuse it, guarded by a file lock so parallel SLURM jobs and concurrent local runs stay consistent.

SUPREME evolved from the codebases of [Selective Synaptic Dampening (SSD)](https://github.com/if-loops/selective-synaptic-dampening) and [bad-teaching unlearning](https://github.com/vikram2000b/bad-teaching-unlearning), generalising them from single-method, single-device scripts into a standardised, distributed evaluation platform.

For the formal pipeline algorithm and mathematical notation (seed formulas, set definitions, operation signatures), see [`src/supreme/README.md`](src/supreme/README.md) and [`docs/notation.md`](docs/notation.md).

---

## 📦 SUPREME as a Library

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
[`src/supreme/__init__.py`](src/supreme/__init__.py); resolution and plugin entry points
live in [`src/supreme/registry.py`](src/supreme/registry.py). Full walkthrough:
[`docs/extending.md`](docs/extending.md) and the notebook
[`notebooks/custom_components.ipynb`](notebooks/custom_components.ipynb).

### Where the code lives

| Path | What's there |
|---|---|
| [`src/supreme/__init__.py`](src/supreme/__init__.py) | Public API surface (`run_*`, `register_*`) |
| [`src/supreme/registry.py`](src/supreme/registry.py) | Name → component resolution and plugin entry points |
| [`src/supreme/methods/unlearning_methods/`](src/supreme/methods/unlearning_methods/) | Unlearning method implementations |
| [`src/supreme/methods/baselines/`](src/supreme/methods/baselines/) | Retrain / Original baselines |
| [`src/supreme/models/`](src/supreme/models/) | ResNet18, ViT |
| [`src/supreme/datasets/datasets.py`](src/supreme/datasets/datasets.py) | The 5 datasets |
| [`src/supreme/eval_metrics/`](src/supreme/eval_metrics/) | The 9 evaluation metrics |
| [`src/supreme/utils/training/train_main.py`](src/supreme/utils/training/train_main.py) | Training-stage entry point (`supreme-train`) |
| [`src/supreme/utils/unlearning/unlearn_main.py`](src/supreme/utils/unlearning/unlearn_main.py) | Unlearn/evaluate entry point (`supreme-unlearn`) |
| [`src/supreme/utils/fabric/`](src/supreme/utils/fabric/) | Lightning Fabric setup (accelerators, precision, distributed strategies) |

---

## 🗃️ Available Components

Registry-based components are **user-extensible** - implement the relevant interface and register the module path, either in-tree or **from your own package** (runtime API or packaging entry points, no edits to SUPREME). See [`docs/extending.md`](docs/extending.md). The components provided via Lightning Fabric cover the supported hardware and execution configurations.

### Registry-based (user-extensible)

| Component | Available implementations |
|---|---|
| **Datasets** | [CIFAR-10](src/supreme/datasets/datasets.py), [CIFAR-20](src/supreme/datasets/datasets.py), [CIFAR-100](src/supreme/datasets/datasets.py), [PinsFaceRecognition](src/supreme/datasets/datasets.py), [Caltech-101](src/supreme/datasets/datasets.py) |
| **Models** | [ResNet18](src/supreme/models/ResNet18.py), [Vision Transformer (ViT)](src/supreme/models/ViT.py) |
| **Baselines** | [Retrain](src/supreme/methods/baselines/retrain.py), [Original](src/supreme/methods/baselines/original.py) |
| **Unlearning methods** | [Fine-Tuning (FT)](src/supreme/methods/unlearning_methods/finetune.py), [Bad Teacher (BadT)](src/supreme/methods/unlearning_methods/bad_teacher.py), [Random Labels (RL)](src/supreme/methods/unlearning_methods/random_labeling.py), [UNSIR](src/supreme/methods/unlearning_methods/unsir.py), [SSD](src/supreme/methods/unlearning_methods/ssd.py), [LFSSD](src/supreme/methods/unlearning_methods/lfssd.py), [ASSD](src/supreme/methods/unlearning_methods/assd.py), [SCRUB](src/supreme/methods/unlearning_methods/scrub.py), [JIT](src/supreme/methods/unlearning_methods/jit.py) |
| **Evaluation metrics** | [Accuracy](src/supreme/eval_metrics/accuracy.py), [Loss/Error](src/supreme/utils/training/training_utils.py), [ZRF](src/supreme/eval_metrics/zrf.py), [Activation Distance](src/supreme/eval_metrics/activation_distance.py), [JS-Divergence](src/supreme/eval_metrics/jsdiv.py), [Layer-wise Distance](src/supreme/eval_metrics/layerwise_distance.py), [Membership Inference Attack](src/supreme/eval_metrics/membership_inference_attack.py), [Completeness](src/supreme/eval_metrics/completeness.py), [Resource Consumption](src/supreme/eval_metrics/resource_consumption.py), [Time](src/supreme/eval_metrics/time.py) |
| **Unlearning scenarios** | Full-class, Subclass, Random sample |

### Provided via Lightning Fabric

| Component | Available implementations |
|---|---|
| **Accelerators** | [CPU](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.CPUAccelerator.html), [CUDA](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.CUDAAccelerator.html), [MPS](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.MPSAccelerator.html), [TPU](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.XLAAccelerator.html) |
| **Precision modes** | [64-true, 32-true, 16-mixed, bf16-mixed, 16-true, bf16-true](https://lightning.ai/docs/fabric/2.1.0/fundamentals/precision.html), [transformer-engine, transformer-engine-float16](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.plugins.precision.TransformerEnginePrecision.html) (FP8), [nf4, nf4-dq, fp4, fp4-dq, int8, int8-training](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.plugins.precision.BitsandbytesPrecision.html) |
| **Distributed strategies** | [DDP](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.DDPStrategy.html), [FSDP](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.FSDPStrategy.html), [DeepSpeed](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.DeepSpeedStrategy.html) (ZeRO Stage 1/2/3) |
| **Loggers** | [Weights & Biases](https://docs.wandb.ai/guides/integrations/lightning), [TensorBoard](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.loggers.TensorBoardLogger.html), [CSV](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.loggers.CSVLogger.html) |

---

## ⚡ Quickstart

```bash
# 1. Clone
git clone https://github.com/pedroandreou/supreme-unlearning.git
cd supreme-unlearning

# 2. Set up environment - the Makefile is the entry point for local dev: it creates
#    the venv (named `unlearning` by default; override with VENV=<name>), installs the
#    pinned deps + SUPREME (editable), and enables the git hook. (Prompts if it
#    already exists; pass ON_EXISTING=reuse to skip.)
make cuda                  # NVIDIA GPU (Linux / WSL2).  Apple Silicon / CPU: `make mps`
source unlearning/bin/activate

# 3. Configure W&B + HF tokens
cp .env.example .env
# edit .env with your WANDB_KEY, WANDB_USERNAME and HUGGING_FACE_HUB_TOKEN

# 4. Smoke test - one seed, one method, one dataset
bash src/supreme/run_local.sh \
  --gpu 0 --models ViT --training-seeds 260 \
  --methods retrain,finetune,ssd \
  --strategies random_ --datasets Cifar10 \
  --forget-percs 0.01
```

Full environment setup (Docker Dev Container, MPS prerequisites, etc.) is documented in [`docs/environment_setup.md`](docs/environment_setup.md). The Docker image is NVIDIA-only (Linux / WSL2); macOS users follow the virtual-env path above.

---

## 🧪 Running Experiments

The pipeline runs **train → unlearn → evaluate** automatically. Re-running is safe: per-stage outputs (training checkpoints, unlearning checkpoints, already-logged W&B results) are detected and skipped.

### Local (workstation, GPU server, interactive cluster node)

```bash
# All 10 seeds, all methods, all datasets - defaults
bash src/supreme/run_local.sh --gpu 0

# Filter the sweep
bash src/supreme/run_local.sh \
  --gpu 0,1 \
  --models ViT \
  --training-seeds 260,261,262 \
  --methods retrain,finetune,bad_teacher,ssd \
  --strategies fullclass,random_ \
  --datasets PinsFaceRecognition
```

| Flag | Description | Default |
|------|-------------|---------|
| `--gpu` | GPU ID(s) - `0` single, `0,1,2,3` multi-GPU | `0` |
| `--models` | `ResNet18`, `ViT` | both |
| `--training-seeds` | Comma-separated training seeds (outer loop, `I`). | `260`–`269` |
| `--unlearning-seeds` | Space-separated indices for `J` (e.g. `"0 1 2"` for `J=3`) | `"0"` (matched) |
| `--evaluation-seeds` | Space-separated indices for `K` | `"0"` (matched) |
| `--methods` | Unlearning methods to run | all 11 (2 baselines + 9 methods) |
| `--strategies` | `fullclass`, `subclass`, `random_` | all |
| `--datasets` | Datasets to use | all 5 |
| `--forget-percs` | Forget % for `random_` strategy | `0.001`–`0.10` |

### SLURM (HPC, login node)

```bash
# Preview the grid (no submission)
./src/supreme/run_slurm.sh --dry-run

# Submit all experiments, max 12 concurrent jobs
./src/supreme/run_slurm.sh --max-concurrent 12

# Subset
./src/supreme/run_slurm.sh \
  --datasets Cifar10,Cifar20 \
  --models ViT \
  --training-seeds 260,261,262

# Multi-GPU DDP per job
./src/supreme/run_slurm.sh --gpus 4
```

Each submitted job runs one `(seed, dataset, model)` cell independently; cells run in parallel across the cluster. Distributed-strategy selection (DDP / FSDP / DeepSpeed) is documented in [`docs/implementation_notes.md → Distributed Strategies`](docs/implementation_notes.md#distributed-strategies).

---

## 🔁 Reproducing the paper

Reproducing the paper's numbers is a two-step process: run the experiment grid on Pins Face Recognition (both architectures, both scenarios, all 10 seeds) and then render the three paper LaTeX tables from the W&B-logged results using [`src/supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb`](src/supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb). The exact command, the table-rendering workflow, and the troubleshooting notes are documented in [`docs/reproducing_the_paper.md`](docs/reproducing_the_paper.md). For a runnable, step-by-step walkthrough (install → smoke test → full grid → tables → extending), see the notebook [`notebooks/reproduce_experiments.ipynb`](notebooks/reproduce_experiments.ipynb).

---

## ➕ Extending SUPREME

SUPREME is reusable as a library (see [SUPREME as a Library](#-supreme-as-a-library)
for installation and the public API). You register your own components from your
own package with no edits to framework code, either at runtime via
`supreme.register_*` or, for an installed plugin package, via packaging entry
points (`supreme.models`, `supreme.unlearning_methods`, `supreme.metrics`,
`supreme.datasets`, `supreme.plugins`).

A runnable, end-to-end walkthrough - `pip install supreme-unlearning`, then
register your own method/metric/model/dataset from your own project - is in
the notebook [`notebooks/custom_components.ipynb`](notebooks/custom_components.ipynb).

Adding a dataset, model, method, or metric follows a consistent register-and-implement pattern. Walkthroughs and Fabric-integration rules live in [`docs/extending.md`](docs/extending.md):

| What to add | Walkthrough |
|---|---|
| New dataset | [`docs/extending.md → Adding a new dataset`](docs/extending.md#adding-a-new-dataset) |
| New model | [`docs/extending.md → Adding a new model`](docs/extending.md#adding-a-new-model) |
| New unlearning method | [`docs/extending.md → Adding a new unlearning method`](docs/extending.md#adding-a-new-unlearning-method) |
| New evaluation metric | [`docs/extending.md → Adding a new evaluation metric`](docs/extending.md#adding-a-new-evaluation-metric) |

---

## 🤝 Contributing

Contributions are welcome - bug reports, new components, and documentation alike.

- **Found a bug or want a feature?** Open an issue - the
  [bug-report and feature-request templates](.github/ISSUE_TEMPLATE) appear
  automatically at
  [New issue → choose a template](https://github.com/pedroandreou/supreme-unlearning/issues/new/choose).
- **Adding a dataset, model, method, or metric?** Most components register from
  your own package with no framework edits - see
  [`docs/extending.md`](docs/extending.md). You can ship it as a `pip`-installable
  plugin or upstream it via a pull request.
- **Opening a pull request?** Run `make style` then `make quality` (the same
  `ruff` lint + format checks CI runs), and follow the
  [PR template](.github/PULL_REQUEST_TEMPLATE.md). Full workflow in the
  [contributing guide](docs/contributing.md).
- **Share your method and results** in [`community/`](community/README.md) and add
  a row to the [leaderboard](community/leaderboard.md).

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) lints, format-checks,
and validates the package build on every push and PR. A version tag like `v0.1.0`
triggers [`.github/workflows/publish.yml`](.github/workflows/publish.yml) to build
and publish the release to PyPI (a manual run targets TestPyPI as a dry-run). The
CUDA images are published to GHCR manually via [`.github/workflows/docker.yml`](.github/workflows/docker.yml)
(runtime image) and [`.github/workflows/devcontainer.yml`](.github/workflows/devcontainer.yml)
(prebuilt dev container). Notable changes per release are tracked in [`CHANGELOG.md`](CHANGELOG.md).

---

## 📚 Documentation

| Document | Covers |
|---|---|
| [`docs/contributing.md`](docs/contributing.md) | How to report issues, add components, and open a pull request |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes per release (Keep a Changelog / SemVer) |
| [`community/`](community/README.md) | Community-contributed methods, templates, and the results leaderboard |
| [`docs/notation.md`](docs/notation.md) | Symbol glossary - seeds, datasets, models, indices, counts |
| [`src/supreme/README.md`](src/supreme/README.md) | Formal algorithm specification (matched and decoupled protocols) |
| [`docs/environment_setup.md`](docs/environment_setup.md) | Virtual-env and Docker Dev Container setup, `.env` template, prerequisites |
| [`docs/reproducing_the_paper.md`](docs/reproducing_the_paper.md) | Single command for the paper's experiment grid plus the W&B-export-to-LaTeX-tables workflow |
| [`docs/script_arguments.md`](docs/script_arguments.md) | Full argument reference for `train_main.py` and `unlearn_main.py` |
| [`docs/extending.md`](docs/extending.md) | How to add new datasets, models, methods, and metrics |
| [`docs/tooling.md`](docs/tooling.md) | Debugger, profiler, Fabric callbacks, process tracker, split export, W&B exporter |
| [`docs/wandb_integration.md`](docs/wandb_integration.md) | W&B runtime behaviour: rank-0 logging, offline mode, sync workflow, metric synchronisation |
| [`docs/wandb_fields.md`](docs/wandb_fields.md) | Paper-to-W&B metric mapping and per-metric field paths |
| [`docs/implementation_notes.md`](docs/implementation_notes.md) | Distributed strategies, gradient handling, batch-size scaling, memory, known limitations |
| [`docs/adding_pinsfacerecognition.md`](docs/adding_pinsfacerecognition.md) | Manual Kaggle download for the Pins Face Recognition dataset |
| [`docs/future_work.md`](docs/future_work.md) | Planned extensions |

---

## 📝 Citing this work

If you use SUPREME in your research, please cite our work. When you use a specific
unlearning method, please also cite its original paper (linked in each method's
source-file header); the foundational SSD/LFSSD and Bad Teacher papers are
included below.

```bibtex
@misc{supreme2026,
  title  = {SUPREME: A Multi-GPU Framework for Reproducible Image Unlearning Method Evaluation},
  author = {Petros Andreou, Jamie Lanyon, Axel Finke, Georgina Cosma},
  year   = {2026},
  eprint = {2606.00380},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2606.00380}
}
@inproceedings{foster2024ssd,
  title     = {Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening},
  author    = {Foster, Jack and Schoepf, Stefan and Brintrup, Alexandra},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024},
  url       = {https://arxiv.org/abs/2308.07707}
}
@inproceedings{foster2024lossfree,
  title     = {Loss-Free Machine Unlearning},
  author    = {Foster, Jack and Schoepf, Stefan and Brintrup, Alexandra},
  booktitle = {ICLR 2024 Tiny Papers Track},
  year      = {2024},
  url       = {https://arxiv.org/abs/2402.19308}
}
@inproceedings{chundawat2023badteacher,
  title     = {Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher},
  author    = {Chundawat, Vikram S and Tarun, Ayush K and Mandal, Murari and Kankanhalli, Mohan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023},
  url       = {https://arxiv.org/abs/2205.08096}
}
```

This work was conducted at [Loughborough University](https://www.lboro.ac.uk/).

---

## 🙏 Acknowledgements

Several unlearning methods reimplement or adapt published research code. We thank
the authors of the following projects, and ask that you cite the original papers
(linked in each method's source-file header) when using the corresponding methods:

- [if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening) - SSD, LFSSD
- [vikram2000b/bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning) - Bad Teacher
- [vikram2000b/Fast-Machine-Unlearning](https://github.com/vikram2000b/Fast-Machine-Unlearning) - UNSIR
- [jwf40/Information-Theoretic-Unlearning](https://github.com/jwf40/Information-Theoretic-Unlearning) - JIT
- [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB) - SCRUB
- [kklusd/Unlearning](https://github.com/kklusd/Unlearning) - NegGrad

---

## 📄 License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file
for details.

---

## ⭐ Star History

<a href="https://star-history.com/#pedroandreou/supreme-unlearning&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pedroandreou/supreme-unlearning&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pedroandreou/supreme-unlearning&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pedroandreou/supreme-unlearning&type=Date" />
  </picture>
</a>
