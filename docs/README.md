# SUPREME documentation

Choose a guide for the task you want to complete.

| Task | Guide |
|---|---|
| Set up and run a first experiment | [Quickstart](quickstart.md) |
| Use the Python API | [Library guide](library.md) |
| Choose built-in components | [Component reference](components.md) |
| Run local or SLURM experiments | [Experiment guide](running_experiments.md) |
| Cite method implementations | [Acknowledgements](acknowledgements.md) |
| Maintain CI and releases | [Maintainer guide](maintaining.md) |
| Explore the technology stack | [Technology and development tools](technology.md) |

## Reference guides

| Document | Covers |
|---|---|
| [Published results](results/README.md) | Existing paper tables, measurement definitions, downloads and the offline viewer |
| [Framework comparison](framework_comparison.md) | Framework capabilities, definitions and sources |
| [`docs/contributing.md`](contributing.md) | How to report issues, add components, and open a pull request |
| [`CHANGELOG.md`](../CHANGELOG.md) | Notable changes per release (Keep a Changelog / SemVer) |
| [`community/`](../community/README.md) | Community-contributed methods, templates, and the results leaderboard |
| [`docs/notation.md`](notation.md) | Symbol glossary - seeds, datasets, models, indices, counts |
| [`docs/seed_protocols.md`](seed_protocols.md) | Why training seeds matter, variance decomposition, and choosing a nested protocol |
| [`src/supreme/README.md`](../src/supreme/README.md) | Formal algorithm specification (matched and decoupled protocols) |
| [`docs/environment_setup.md`](environment_setup.md) | Virtual-env and Docker Dev Container setup, `.env` template, prerequisites |
| [`docs/reproducing_the_paper.md`](reproducing_the_paper.md) | Single command for the paper's experiment grid plus the W&B-export-to-LaTeX-tables workflow |
| [`docs/script_arguments.md`](script_arguments.md) | Full argument reference for `train_main.py` and `unlearn_main.py` |
| [`docs/extending.md`](extending.md) | How to add new datasets, models, methods, and metrics |
| [`docs/tooling.md`](tooling.md) | Debugger, profiler, Fabric callbacks, process tracker, split export, W&B exporter |
| [`docs/wandb_integration.md`](wandb_integration.md) | W&B runtime behaviour: rank-0 logging, offline mode, sync workflow, metric synchronisation |
| [`docs/wandb_fields.md`](wandb_fields.md) | Paper-to-W&B metric mapping and per-metric field paths |
| [`docs/implementation_notes.md`](implementation_notes.md) | Distributed strategies, gradient handling, batch-size scaling, memory, known limitations |
| [`docs/adding_pinsfacerecognition.md`](adding_pinsfacerecognition.md) | Manual Kaggle download for the Pins Face Recognition dataset |
| [`docs/future_work.md`](future_work.md) | Planned extensions |
