# W&B Utilities

Three stages, one per directory:

```
wandb_utils/
├── runtime/              Stage 0: live W&B logging during training/unlearning runs
│   ├── wandb_setup.py        - initialise wandb runs, sync, check existence (imported by train_main.py / unlearn_main.py)
│   ├── wandb_manager.py      - CLI for managing runs (list/dedupe/cleanup/missing-run audit)
│   └── wandb_tools.sh        - shell wrapper around wandb_manager.py
│
├── results_extraction/   Stage 1: pull experiment results from the W&B server to local CSVs
│   ├── orchestrate_wandb_export.sh   - main entry point
│   ├── export_config.py              - shared constants (seeds, prefixes, models, datasets, etc.)
│   └── wandb_metrics_summary/        - generated CSVs (gitignored)
│
└── results_analysis/     Stage 2: aggregate the CSVs and emit the LaTeX tables for the paper
    └── results_tables.ipynb
```

## Stage 0 - Runtime logging

`wandb_setup.py` is imported automatically by `train_main.py` and `unlearn_main.py` (it initialises the W&B run, syncs at the end, and skips already-logged runs). Nothing for you to run directly.

`wandb_tools.sh` is the user-facing shell entry point. It covers two things:

1. **Push local offline runs to the W&B server** (the `sync` mode - bash, implemented in this script):

   ```bash
   bash src/utils/wandb_utils/runtime/wandb_tools.sh sync                    # basic
   bash src/utils/wandb_utils/runtime/wandb_tools.sh sync --delete-synced    # drop local dirs after a successful sync
   bash src/utils/wandb_utils/runtime/wandb_tools.sh sync --delete-immediately
   ```

2. **Server-side housekeeping** - listing projects, finding/deleting duplicate runs, cleaning empty runs, finding missing seeds, generating duplicate reports. These modes forward to `wandb_manager.py`:

   ```bash
   bash src/utils/wandb_utils/runtime/wandb_tools.sh list-projects
   bash src/utils/wandb_utils/runtime/wandb_tools.sh find-duplicates --all
   bash src/utils/wandb_utils/runtime/wandb_tools.sh cleanup-empty --project "<NAME>" --delete
   ```

Run `wandb_tools.sh` with no args to see the full list of modes.

## Stage 1 - Extract results from W&B

```bash
# Process the curated set of project combinations (recommended)
bash src/utils/wandb_utils/results_extraction/orchestrate_wandb_export.sh --all-existing
```

Output lands under `results_extraction/wandb_metrics_summary/` (one subfolder per dataset, one CSV per `<model>_<dataset>_<strategy>_<forget_target>` project).

The flags are:

| Flag | Action |
|------|--------|
| (default) | Run the full pipeline: export → combine → analyze |
| `--export` | Only export metrics from W&B |
| `--combine` | Only combine exported data |
| `--analyze` | Only generate visualizations |
| `--clean` | Delete all generated directories and logs |

You can also process a single combination by passing positional args:

```bash
bash src/utils/wandb_utils/results_extraction/orchestrate_wandb_export.sh ResNet18 Cifar20 fullclass 32-true 0
```

All shared constants (seeds, experiment prefixes, models, dataset/strategy combinations) live in `results_extraction/export_config.py`. Edit there to change defaults.

## Stage 2 - Generate LaTeX tables

Open the notebook and run all cells:

```bash
jupyter notebook src/utils/wandb_utils/results_analysis/results_tables.ipynb
```

The notebook reads the CSVs from `results_extraction/wandb_metrics_summary/`, averages across forget targets per scenario, computes mean ± std across the 10 seeds, and emits one LaTeX table per dataset.
