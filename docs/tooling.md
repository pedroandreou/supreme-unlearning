# Tooling

Development and runtime utilities shipped with SUPREME.

- [Debugger (VS Code)](#debugger-vs-code)
- [Profiler (Scalene)](#profiler-scalene)
- [Fabric callbacks](#fabric-callbacks)
- [Process auto-cleanup](#process-auto-cleanup)
- [Forget / retain split export](#forget--retain-split-export)
- [W&B metrics exporter](#wb-metrics-exporter)

---

## Debugger (VS Code)

Enable advanced debugging (CUDA/MPS error tracing, blocking calls) by setting `DEBUGGER=1`:

```bash
# Standalone
DEBUGGER=1 bash supreme/run_local.sh --gpu 0 --methods finetune --models ViT --training-seeds 260

# In .vscode/tasks.json (Docker Dev Container)
"args": ["-c", "cd /app/host/src && DEBUGGER=1 bash run_local.sh --gpu 0 --methods finetune"]
```

**How it works.** `debugpy` attaches to rank 0 (port 5678, listens on 0.0.0.0) after distributed setup but before training. Other ranks (> 0) continue normally. Set breakpoints in VS Code; rank 0 pauses.

**Implementation:** attachment in [`supreme/utils/training/train_main.py`](../supreme/utils/training/train_main.py) and [`supreme/utils/unlearning/unlearn_main.py`](../supreme/utils/unlearning/unlearn_main.py); core logic in [`supreme/utils/debug_utils.py`](../supreme/utils/debug_utils.py) (`create_debugger_session()`).

Debugging flags are only enabled when `DEBUGGER=1`, keeping normal runs fast.

---

## Profiler (Scalene)

Enable line-level CPU / GPU / memory profiling with [Scalene](https://github.com/plasma-umass/scalene) by setting `SCALENE=1`:

```bash
# Standalone
SCALENE=1 ./run_local.sh ...

# In .vscode/tasks.json
"args": ["-c", "cd /app/host/src && SCALENE=1 bash run_local.sh --gpu 0 --methods finetune"]
```

When `SCALENE=1`, the framework automatically sets `num_workers=0` for all DataLoaders (see [`supreme/utils/generic_utils.py`](../supreme/utils/generic_utils.py)) because Scalene has limited multiprocessing support. `SCALENE` can be combined with `DEBUGGER` for simultaneous profiling and debugging.

---

## Fabric callbacks

SUPREME uses [Lightning Fabric callbacks](https://lightning.ai/docs/fabric/2.1.0/guide/callbacks.html) to hook into the training/unlearning/evaluation loops without polluting the main code. Register callback classes in [`supreme/utils/fabric/callbacks.py`](../supreme/utils/fabric/callbacks.py) and call their hooks:

```python
fabric.call("on_train_epoch_start", fabric=fabric, epoch=epoch)
fabric.call("on_train_batch_start")
fabric.call("on_train_batch_end", loss=loss, epoch=epoch, batch_idx=batch_idx, lr=current_lr)
fabric.call("on_train_epoch_end", epoch=epoch, train_loss=train_loss, last_lr=lrs[-1])
```

Provided callbacks: `TrainingCallback` (training/unlearning), `TestCallback` (validation/testing), plus parameter-modification and metric callbacks.

**When to use.** Callbacks are great for debugging and developing new methods/metrics - they reveal how metrics, losses, and timings change at every step.

**When they're disabled.** During unlearning and metric evaluation the callback calls are commented out so the logging overhead doesn't skew our timing, power, and memory measurements. Initial training keeps them enabled (we don't benchmark training).

When callbacks are enabled, the framework logs every 10 batches (see `if batch_idx % 10 == 0:` in [`supreme/utils/fabric/callbacks.py`](../supreme/utils/fabric/callbacks.py)). The regular logging also feeds the [process tracker](#process-auto-cleanup) so it can distinguish healthy progress from stalls.

W&B specifics - rank-0 logging, offline mode, sync workflow, and metric synchronisation - are documented separately in [`docs/wandb_integration.md`](wandb_integration.md).

---

## Process auto-cleanup

Distributed frameworks (PyTorch Lightning, Lightning Fabric) provide built-in process management via `init_process_group()`, `destroy_process_group()`, `setup()`, `teardown()`. These only guarantee cleanup on **successful** completion. If a process crashes, hangs at a barrier, or NCCL fails to release CUDA resources, orphan "zombie" processes can remain and tie up GPU memory or block future jobs.

The SUPREME process tracker is the safety net.

**What it does:**

- Assigns each cell a unique experiment ID (script type, model, dataset, timestamp)
- Watches log files for activity and detects stalled processes (180 s of log inactivity + no completion marker)
- Force-terminates stalled processes
- Handles `SIGINT` (Ctrl+C) and `SIGTERM` gracefully

**Best practice.** Run the tracker alongside frequent logging via [callbacks](#fabric-callbacks) - without log output, healthy processes can be misidentified as stalled (false positives).

**Note on benchmarking.** The tracker was disabled during the paper's final runs so that reported execution times reflect raw runtime, without monitoring overhead. Re-enable it for development and for any production / multi-day run.

---

## Forget / retain split export

When dataset-distribution export is enabled, SUPREME writes per-class composition of every forget/retain split.

For each forget class (e.g. `people`, `veg`) two directories are produced:

```
logs/dataset_distributions/<class_name>/train/
logs/dataset_distributions/<class_name>/test/
```

Each contains two subdirectories:

- `class_distribution/` - `forget_class_distribution.csv` and `retain_class_distribution.csv` with columns `Class Type, Class Name, Count`:
  ```
  Class Type,Class Name,Count
  Class,Unknown Class,7500
  Class,veg,500
  Class,electrical_devices,500
  ...
  ```
  `Unknown Class` means the integer label wasn't registered in [`supreme/utils/project_config.py`](../supreme/utils/project_config.py) (e.g. `cifar20_dict`, `cifar100_dict`, `pins_dict`) - we only define labels for classes we care about for forgetting. Reference: original mappings in [bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning).

- `set_info/` - `forget_set_info.csv` and `retain_set_info.csv` with detailed batch/sample info:
  ```
  Batch,Index,Superclass,Subclass
  0,0,people,boy
  0,1,people,baby
  0,2,people,girl
  ```

Useful for verifying splits, catching data leakage, and debugging unexpected per-class results. Implementation in [`supreme/utils/debug_utils.py`](../supreme/utils/debug_utils.py).

---

## W&B metrics exporter

Fetch, combine, and visualise metrics from W&B projects via the orchestrator script:

```bash
# Full pipeline (export → combine → analyse) for curated projects (recommended)
./supreme/utils/wandb_utils/results_extraction/orchestrate_wandb_export.sh --all-existing

# Single combination
./supreme/utils/wandb_utils/results_extraction/orchestrate_wandb_export.sh --export ResNet18 Cifar20 fullclass veg

# Clean generated files
./supreme/utils/wandb_utils/results_extraction/orchestrate_wandb_export.sh --clean
```

### Execution modes

| Mode | Description |
|------|-------------|
| `--all-existing` | Curated project combinations (fastest, recommended) |
| `--all-possible` | All possible combinations (slower, may hit non-existent projects) |
| Positional args | Single combination, e.g. `ResNet18 Cifar20 fullclass veg` |

### Pipeline steps

| Flag | Action |
|------|--------|
| *(default)* | Full pipeline: export → combine → analyse |
| `--export`  | Only export metrics from W&B |
| `--combine` | Only combine exported data |
| `--analyze` | Only generate visualisations |
| `--clean`   | Delete generated directories and logs |

Two analysis notebooks consume the exported CSVs:

- [`supreme/utils/wandb_utils/results_analysis/all_results_exploration.ipynb`](../supreme/utils/wandb_utils/results_analysis/all_results_exploration.ipynb) - broad multi-dataset analysis across the whole sweep.
- [`supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb`](../supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb) - produces the exact LaTeX tables in the paper (Pins Face Recognition main + appendix).

For the field-naming conventions used by the exporter (paper-to-W&B metric mapping, per-metric paths), see [`docs/wandb_fields.md`](wandb_fields.md). For W&B's runtime behaviour (rank-0 logging, offline mode, sync workflow), see [`docs/wandb_integration.md`](wandb_integration.md).
