# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-08-18

### Added

- Deterministic SSD/LFSSD variants (SSD-Det, LFSSD-Det), with the JIT docstring
  aligned to match.
- Training-stage W&B logging toggle and clearer per-phase progress output.

### Changed

- Corrected the SSD/LFSSD citations for the `-det` variants.
- README: publication badge updated from "Under Review" to "Published at the
  WIPE-OUT 2 Workshop (ECML-PKDD 2026)".
- Docs: "unlearning scenario" is now used consistently for
  fullclass/subclass/random_ (disambiguated from Lightning's distributed
  strategies), and the former "2 baselines" wording now describes Retrain as the
  gold-standard baseline and Original as the unmodified reference.
- `docker/docker-compose.yml`: the `cuda_12_1_pulled` service now points at the
  `0.1.4` runtime image (was `0.1.2`).

### Security

- `SECURITY.md`: documented the dependency-pinning policy - the stack stays
  frozen to the paper-reproduction environment, and dependency advisories that
  require loading untrusted checkpoints/models are dismissed as tolerable risk,
  with safe-usage guidance for users.

## [0.1.3] - 2026-06-03

### Fixed

- `make cuda` and `pip install supreme-unlearning[cuda]` no longer fail on hosts
  that have the NVIDIA driver but no CUDA toolkit. DeepSpeed was compiling CUDA ops
  at install time and aborting with `CUDA_HOME does not exist`; it is now an opt-in
  extra instead of a default CUDA dependency.

### Changed

- DeepSpeed split out of the `[cuda]` extra into its own `[deepspeed]` extra (the
  `[cuda]` extra now installs only wheel-based packages and is safe on any NVIDIA
  host). Opt in with `pip install supreme-unlearning[deepspeed]`, `make deepspeed`,
  or `requirements/requirements.deepspeed.txt` once a CUDA toolkit (`nvcc`/
  `CUDA_HOME`) is available. The CUDA-devel Docker image still ships DeepSpeed.

### Added

- `docs/environment_setup.md`: note that the NVIDIA Container Toolkit is a
  host-daemon prerequisite for GPU-in-Docker (and that the native venv needs no
  toolkit), plus a `torch.cuda` GPU self-check snippet.

## [0.1.2] - 2026-06-02

### Added

- CPU-only pytest suite covering the registry resolution contract, eval-metric
  math, model forward shapes, and an import smoke test over every module, wired
  into CI as a dedicated `tests` job alongside lint and build.
- CI status badge in the README.

### Changed

- The `supreme` package now lives under `src/` (src-layout). The import name
  (`import supreme`) and distribution name (`supreme-unlearning`) are unchanged
  and the built wheel is identical; only the on-disk source location differs.
- Committed notebooks now retain their outputs: removed the `nbstripout`
  pre-commit hook added in 0.1.1.

## [0.1.1] - 2026-06-02

Documentation, packaging, and developer-tooling release. No functional changes
to the `supreme` package.

### Added

- Prebuilt dev container image on GHCR (`supreme-unlearning-devcontainer`);
  "Reopen in Container" now pulls a ready-made image instead of building locally.
- `nbstripout` pre-commit hook so committed notebooks stay output-free.
- Security policy (`.github/SECURITY.md`) with private vulnerability reporting.

### Fixed

- CUDA Docker image now builds: pinned the Python 3.9 `get-pip` bootstrap URL and
  corrected stale `/app/host/src` paths left from the `src -> supreme` rename.
- Stale documentation references: the dead "Open in Dev Containers" badge link, an
  incorrect claim that a version tag publishes the Docker image to GHCR, wrong
  function/line-anchor references, the wandb results notebook name, and
  undocumented logging CLI flags in `docs/script_arguments.md`.
- PyPI version badge no longer shows "package or version not found" (Camo cache bust).

### Changed

- README overhaul and tagline rebrand to "A Multi-GPU Framework for Reproducible
  Image Unlearning Method Evaluation" (wordmark SVG and citations updated to match).
- Consolidated lint/build recipes into the Makefile as the single source of truth,
  mirrored by CI, the Docker image, and the dev container.

## [0.1.0] - 2026-06-02

First public release of **SUPREME** - a registry-based, multi-GPU framework for
reproducible image-unlearning evaluation.

Install with `pip install supreme-unlearning` (import as `supreme`). Console
scripts: `supreme-train`, `supreme-unlearn`. Pin paper reproduction to the
`v0.1.0-paper` reference tag.

### Added
- **Pip-installable distribution** `supreme-unlearning`: full `pyproject.toml`
  metadata, dependency pins (PyTorch + Lightning stack), extras (`[cuda]`,
  `[tensorboard]`, `[dev]`), dynamic version, console scripts and entry-point
  groups for plugins. `setup.py` reduced to a compatibility shim.
- **Public API** (`src/supreme/__init__.py`, torch-free so registration needs no GPU
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
- **Registry** extended additively - resolution order is runtime overrides →
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

[0.1.4]: https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.4
[0.1.3]: https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.3
[0.1.2]: https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.2
[0.1.1]: https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.1
[0.1.0]: https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.0
