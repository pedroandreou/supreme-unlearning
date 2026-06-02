# W&B Integration

SUPREME logs evaluation results to [Weights & Biases](https://wandb.ai/) via [Lightning Fabric's `WandbLogger`](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.pytorch.loggers.wandb.html). This document covers the runtime behaviour. For the field-naming conventions used by the exporter (paper-to-W&B metric mapping), see [`docs/wandb_fields.md`](wandb_fields.md). For the exporter CLI itself, see [`docs/tooling.md → W&B metrics exporter`](tooling.md#wb-metrics-exporter).

## Rank-0 logging

In a multi-GPU run, only rank 0 writes to W&B - duplicates from other ranks are suppressed automatically by Lightning's `WandbLogger`. No SUPREME-specific code is required; you don't need to guard logging calls with `if fabric.global_rank == 0:` for the logger itself (you *do* need such guards for random-tensor generation inside unlearning methods - see [`docs/extending.md → Distributed synchronisation`](extending.md#4-distributed-synchronisation-for-random-operations)).

## Online vs offline mode

W&B runs in **online mode by default**. To switch to offline mode, uncomment this line in [`src/supreme/utils/wandb_utils/runtime/wandb_setup.py`](../src/supreme/utils/wandb_utils/runtime/wandb_setup.py):

```python
# os.environ["WANDB_MODE"] = "offline"
```

Offline mode is useful on air-gapped clusters or to avoid network flakiness during long runs. Offline runs are stored under `wandb/` and can be uploaded later with the sync workflow below.

## Sync workflow (for offline runs)

Batch-upload previously-saved offline runs to the W&B cloud:

```bash
# Sync all pending offline runs
src/supreme/utils/wandb_utils/runtime/wandb_tools.sh sync

# Sync + clean local directories after each successful upload
src/supreme/utils/wandb_utils/runtime/wandb_tools.sh sync --delete-immediately

# Sync + clean local directories only after ALL syncs complete
src/supreme/utils/wandb_utils/runtime/wandb_tools.sh sync --delete-synced

# Treat conflicted runs as new runs (force-resolve conflicts)
src/supreme/utils/wandb_utils/runtime/wandb_tools.sh sync --clean
```

The orchestrator behind these commands lives in [`src/supreme/utils/wandb_utils/runtime/wandb_tools.sh`](../src/supreme/utils/wandb_utils/runtime/wandb_tools.sh). See the script header for the full sync-mode flag list.

## Metric synchronisation across processes

Evaluation metrics that are computed independently on each rank are aggregated before logging. The pattern is `fabric.all_gather()` inside each metric module, followed by `.mean()` / `.max()` / `.sum()` as appropriate.

Examples in the codebase:

- [`src/supreme/eval_metrics/jsdiv.py`](../src/supreme/eval_metrics/jsdiv.py) - `fabric.all_gather(local_js_div)` then `.mean()`
- [`src/supreme/eval_metrics/membership_inference_attack.py`](../src/supreme/eval_metrics/membership_inference_attack.py) - `fabric.all_gather()` on the forget / retain features and labels before the attack model is trained
- [`src/supreme/eval_metrics/accuracy.py`](../src/supreme/eval_metrics/accuracy.py) - `fabric.all_gather()` on per-rank correct-count totals

For implementing a new metric, the same pattern is documented in [`docs/extending.md → Adding a new evaluation metric`](extending.md#adding-a-new-evaluation-metric).

## Skipping already-logged results

Before running a cell, [`src/supreme/MAIN.sh`](../src/supreme/MAIN.sh) checks whether the corresponding W&B run already exists with all requested evaluation metrics logged. If yes, the cell is skipped; partial results trigger a re-run for only the missing metrics. The check is implemented in [`check_wandb_run_exists()`](../src/supreme/utils/wandb_utils/runtime/wandb_setup.py) (see `wandb_setup.py`). Override the skip behaviour with `--force-rerun` (SLURM) or `FORCE_REUNLEARNING=true FORCE_REEVALUATION=true` (env vars).

## Run-name convention

W&B run names encode the seed protocol so the export pipeline can group runs correctly:

| Protocol | Run name pattern |
|---|---|
| Matched (`J = K = 1`) | `{method}_seed{S}` (legacy) or `{method}_tseed{T}_useed{U}` when `TRAINING_SEED` is set |
| Decoupled `I × J` (`J > 1`, `K = 1`) | `{method}_tseed{T}_useed{U}` |
| Decoupled `I × J × K` (`J > 1`, `K > 1`) | `{method}_tseed{T}_useed{U}_eseed{E}` |

Names are generated in [`src/supreme/utils/wandb_utils/runtime/wandb_setup.py`](../src/supreme/utils/wandb_utils/runtime/wandb_setup.py) and [`src/supreme/utils/unlearning/unlearn_main.py`](../src/supreme/utils/unlearning/unlearn_main.py). The full seed-protocol notation lives in [`docs/notation.md`](notation.md).
