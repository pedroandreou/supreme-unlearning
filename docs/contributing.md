# Contributing to SUPREME

Everyone is welcome to contribute, and every contribution is valued, not just
code. Answering questions, helping others, improving documentation, and
reporting bugs are all appreciated. If you find SUPREME useful, please ⭐️ the
repo, cite it, and share it.

> 🤝 The structure of this guide is inspired by the
> [Transformers contributing guide](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

## Ways to contribute

- Fix bugs in the existing code.
- Report bugs or request features (see below).
- Add a new **dataset, model, unlearning method, evaluation metric, or scenario**.
- Improve the documentation.

## Reporting bugs and requesting features

Please use the issue templates. They appear automatically when you
[open a new issue](https://github.com/pedroandreou/supreme-unlearning/issues/new/choose):

- **🐛 Bug report**: first search existing issues, then give the **exact command
  or `supreme.run_*` call**, the full traceback, and your environment (OS,
  install method, accelerator, Python/PyTorch versions).
- **🚀 Feature request**: describe the feature, the motivation, and (if it
  relates to a paper) a link. Note that most components can be added without any
  framework changes (see below), so check that path first.

## Adding a new component

SUPREME is **registry-based**, so adding a dataset, model, unlearning method, or
metric means implementing a small interface and registering its module path, with no
changes to framework internals. The walkthroughs live in
[`docs/extending.md`](extending.md):

| What to add | Walkthrough |
|---|---|
| New dataset | [Adding a new dataset](extending.md#adding-a-new-dataset) |
| New model | [Adding a new model](extending.md#adding-a-new-model) |
| New unlearning method | [Adding a new unlearning method](extending.md#adding-a-new-unlearning-method) |
| New evaluation metric | [Adding a new evaluation metric](extending.md#adding-a-new-evaluation-metric) |

There are **two ways** to share a component, and you can choose either:

1. **As your own plugin package (no PR needed).** SUPREME is pip-installable
   (`pip install supreme-unlearning`) and resolves components from external
   packages via the runtime API (`supreme.register_*`) or packaging entry points
   (`supreme.models`, `supreme.unlearning_methods`, `supreme.metrics`,
   `supreme.plugins`). Publish your package and others can `pip install` it and
   use your component immediately. See
   [`notebooks/custom_components.ipynb`](../notebooks/custom_components.ipynb) for
   an end-to-end example, and list it in [`community/`](../community/README.md) so
   others can find it.

2. **In-tree, via a pull request.** If your component is broadly useful and you'd
   like it shipped with SUPREME, open a PR adding it under `supreme/` with a short
   `community/` entry documenting it (see the
   [method template](../community/methods/template/README.md)).

## Opening a pull request

1. Fork the repo and create a branch off `main`.
2. Set up the dev environment and the git hook: `make cuda` (NVIDIA) or `make mps`
   (Apple Silicon / CPU). This installs SUPREME editable and enables the
   pre-commit hook.
3. Make your change. Keep it focused; add or update docs where usage changes.
4. Run the quality checks before pushing:
   ```bash
   make style     # auto-fix lint + format in place
   make quality   # CI-style check (ruff lint + format, no edits); this is what CI runs
   ```
5. Push and open a PR. The PR template will guide you through the checklist.

CI (`.github/workflows/ci.yml`) runs the same `ruff` lint + format checks and a
packaging build check on every PR, so a green `make quality` locally should mean
a green CI.

## Documentation

When you change behaviour or add a component, update the relevant doc under
[`docs/`](.) and, if user-facing, the [README](../README.md). Documentation-only
PRs are very welcome and can skip the component checklist.
