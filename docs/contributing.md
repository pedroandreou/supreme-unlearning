# Contributing to SUPREME

Everyone is welcome to contribute, and every contribution is valued, not just
code. Answering questions, helping others, improving documentation, and
reporting bugs are all appreciated. If you find SUPREME useful, please ⭐️ the
repo, cite it, and share it.

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
[`docs/extending.md`](https://github.com/pedroandreou/supreme-unlearning/blob/main/docs/extending.md):

| What to add | Walkthrough |
|---|---|
| New dataset | [Adding a new dataset](https://github.com/pedroandreou/supreme-unlearning/blob/main/docs/extending.md#adding-a-new-dataset) |
| New model | [Adding a new model](https://github.com/pedroandreou/supreme-unlearning/blob/main/docs/extending.md#adding-a-new-model) |
| New unlearning method | [Adding a new unlearning method](https://github.com/pedroandreou/supreme-unlearning/blob/main/docs/extending.md#adding-a-new-unlearning-method) |
| New evaluation metric | [Adding a new evaluation metric](https://github.com/pedroandreou/supreme-unlearning/blob/main/docs/extending.md#adding-a-new-evaluation-metric) |

There are **two ways** to share a component, and you can choose either:

1. **As your own plugin package (no PR needed).** SUPREME is pip-installable
   (`pip install supreme-unlearning`) and resolves components from external
   packages via the runtime API (`supreme.register_*`) or packaging entry points
   (`supreme.models`, `supreme.unlearning_methods`, `supreme.metrics`,
   `supreme.plugins`). Publish your package and others can `pip install` it and
   use your component immediately. See
   [`notebooks/custom_components.ipynb`](https://github.com/pedroandreou/supreme-unlearning/blob/main/notebooks/custom_components.ipynb) for
   an end-to-end example, and list it in [`community/`](https://github.com/pedroandreou/supreme-unlearning/blob/main/community/README.md) so
   others can find it.

2. **In-tree, via a pull request.** If your component is broadly useful and you'd
   like it shipped with SUPREME, open a PR adding it under `src/supreme/` with a short
   `community/` entry documenting it (see the
   [method template](https://github.com/pedroandreou/supreme-unlearning/blob/main/community/methods/template/README.md)).

## File header convention

Every component file (datasets, models, methods, baselines, metrics) starts with
a **module docstring on the first line**, before the imports. It carries a
one-line summary and, for any code derived from a paper or another codebase, the
attribution. Use these labels so headers stay consistent and grep-able:

```python
"""<One-line summary of the component>.

Paper: "<Title>" (<paper-url>)
Reference: <upstream implementation url>

Notes:
<adaptation details, validation against the original, or naming caveats>
"""

import ...
```

- **`Paper:`** - the source paper, title in quotes with the URL in parentheses.
  Repeat the `Paper:` / `Reference:` pair (separated by a blank line) when a file
  draws on more than one source.
- **`Reference:`** - a link to the upstream implementation you adapted, pinned to
  a specific commit and line where possible (e.g. `.../blob/<sha>/src/ssd.py#L35`).
  Omit it for fully original code.
- **`Notes:`** - anything else worth preserving: what you changed from the
  original, how you validated the port, or alternative implementations you
  considered. Keep every upstream URL; don't drop attribution when editing.

If your component derives from third-party code, keep its attribution in the
source-file header (`Paper:`, `Reference:`, `Notes:`) and credit the upstream
project in the README [Acknowledgements](https://github.com/pedroandreou/supreme-unlearning#-acknowledgements)
section.

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
[`docs/`](https://github.com/pedroandreou/supreme-unlearning/tree/main/docs) and, if user-facing, the [README](https://github.com/pedroandreou/supreme-unlearning/blob/main/README.md). Documentation-only
PRs are very welcome and can skip the component checklist.
