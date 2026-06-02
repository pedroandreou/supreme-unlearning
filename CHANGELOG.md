# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-02

First public release of **SUPREME** — a registry-based, multi-GPU framework for
reproducible image-unlearning evaluation.

Install with `pip install supreme-unlearning` (import as `supreme`). Console
scripts: `supreme-train`, `supreme-unlearn`. Pin paper reproduction to the
`v0.1.0-paper` reference tag.

### Added
- **Pip-installable distribution** `supreme-unlearning`: full `pyproject.toml`
  metadata, dependency pins (PyTorch + Lightning stack), extras (`[cuda]`,
  `[tensorboard]`, `[dev]`), dynamic version, console scripts and entry-point
  groups for plugins. `setup.py` reduced to a compatibility shim.
- **Public API** (`supreme/__init__.py`, torch-free so registration needs no GPU
  stack): `register_model`, `register_baseline`, `register_unlearning_method`,
  `register_metric`, `register_dataset`, `run_training`, `run_unlearning`,
  `project_config`, `__version__`.
- **External extensibility**: register your own unlearning methods, metrics,
  models and datasets via the runtime API or packaging entry points, with no
  edits to framework code. `register_*` also accepts a live callable.
- **Makefile** as the single entry point (venv / deps / editable install / build /
  publish).
- **CI** (GitHub Actions): ruff + build check; tag → PyPI via trusted publishing
  + GitHub Release; manual `ghcr.io` Docker build.
- **Docs & notebooks**: runnable `reproduce_experiments` and `custom_components`
  notebooks; docs for pip-install, public API, console scripts and the
  `v0.1.0-paper` reference tag.
- Optional `SUPREME_PROJECT_ROOT` override (default unchanged).
- macOS portability: portable training lock (flock on Linux, mkdir spinlock on
  macOS), venv auto-activate probe, ViT loads from the HF cache.

### Changed
- **Renamed package `src` → `supreme`** (mechanical, behaviour-identical): all
  imports, registry module-path strings, bash launch paths, Docker paths and docs
  updated; external citation URLs preserved.
- **Registry** extended additively — resolution order is runtime overrides →
  entry points → built-in convention; built-in resolution is unchanged.
- Renamed the resource metric `power_consumption` → `compute_utilisation` (the
  paper's term) across producer and consumers.

### Removed
- Legacy nvidia-smi `power.draw` fallback (it measured power, not utilisation);
  pynvml smUtil and the MPS sampling path are retained.
- Dead `SAMPLE_SCALING` / `SCALABLE_EXPERIMENT_SCENARIO` experiment knobs.

### Fixed
- `compute_utilisation`: `end_compute_util` was aliasing the run average on both
  the NVML (NVIDIA) and MPS (Apple Silicon) paths. It now reports a genuine
  end-of-run reading (last sample, falling back to the start snapshot), symmetric
  with `start_compute_util`.
- Repo hygiene: stale `src/` paths in `.gitignore` (which had exposed ~17k
  untracked dataset images), anchored venv ignores, `.dockerignore` /
  `.vscode/tasks.json` paths, and added `SLURM_ACCOUNT` to `.env.example`.

[0.1.0]: https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.0
