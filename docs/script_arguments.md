# Advanced: Direct Script Arguments

**Note:** Most users should use [run_local.sh](../supreme/run_local.sh) (documented in the README). This section is for advanced users running [train_main.py](../supreme/utils/training/train_main.py) or [unlearn_main.py](../supreme/utils/unlearning/unlearn_main.py) directly.

The two stages can be invoked three equivalent ways, all taking the same
arguments documented below:

- the scripts directly: `python supreme/utils/training/train_main.py ...`
- the installed console scripts: `supreme-train ...` / `supreme-unlearn ...`
- the Python API: `supreme.run_training([...])` / `supreme.run_unlearning([...])`

## Common Arguments (both scripts)

Defined in [common_args.py](../supreme/utils/parsers/common_args.py):

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-net` | str | required | Model architecture (`ResNet18`, `ViT`) |
| `-lr` | float | `0.1` | Initial learning rate |
| `-warm` | int | `1` | Warm-up training phase duration (epochs) |
| `-distributed_strategy` | str | `"ddp"` | Distributed training strategy (or `DISTRIBUTED_STRATEGY` env var). Choices: `ddp`, `fsdp`, `deepspeed`, `auto`, `xla` |
| `-deepspeed_stage` | int | `2` | DeepSpeed ZeRO stage (or `DEEPSPEED_STAGE` env var; only used when `-distributed_strategy=deepspeed`). Choices: `1`, `2`, `3` |
| `-wandb_logging_flag` | flag | `False` | Enable W&B logging |
| `-tensorboard_logging_flag` | flag | `False` | Enable Lightning Fabric TensorBoardLogger (requires `tensorboard`/`tensorboardX`) |
| `-csv_logging_flag` | flag | `False` | Enable Lightning Fabric CSVLogger (writes `metrics.csv` under `-logging_root_dir`) |
| `-logging_root_dir` | str | `"./fabric_logs"` | Root directory for Fabric CSV/TensorBoard loggers (or `FABRIC_LOGGING_ROOT_DIR` env var) |
| `-export_class_distribution_info_flag` | flag | `False` | Export class distribution CSVs |
| `-use_process_tracker` | flag | `False` | Enable zombie process monitoring |

## Training ([train_main.py](../supreme/utils/training/train_main.py))

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-precision` | str | required | Training precision (`32-true`, `bf16-mixed`, `16-mixed`, etc.) |
| `-dataset` | str | required | Dataset name (from `project_config.dataset_names`) |
| `-classes` | int | required | Number of classes |
| `-batch_size` | int | `64` | Batch size for dataloader |
| `-training_seed` | int | `None` | Seed for reproducible training |
| `-unlearning_seed` | int | `None` | Seed for data splitting/reproducibility |
| `-unlearning_context` | str | `"N/A"` | Description of the unlearning context for this training run |
| `-include_gpus_in_path` | str | `"true"` | Include GPU count in checkpoint path |

## Unlearning ([unlearn_main.py](../supreme/utils/unlearning/unlearn_main.py))

### Always required

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-weight_path` | str | required | Path to trained model checkpoint |
| `-method` | str | required | Unlearning method name (from `project_config.all_methods`) |
| `-precision` | str | required | Precision mode |
| `-eval_metrics` | str | required | Comma-separated metrics (e.g., `accuracy,zrf,mia`) |
| `-type_of_unlearning_strategy` | str | required | Unlearning strategy: `fullclass`, `subclass`, or `random_`. Read by an initial parser before strategy-specific args are added. |
| `-seed` | int | `0` | Random seed |
| `-epochs` | int | `1` | Number of unlearning epochs |

### Strategy-specific arguments

Determined by the `-type_of_unlearning_strategy` flag:

**fullclass:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-classes` | int | required | Number of classes |
| `-batch_size` | int | `64` | Batch size |
| `-forget_class_name` | str | required | Class to forget (from dataset's class dictionary) |

**subclass:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-superclasses` | int | required | Number of superclasses |
| `-subclasses` | int | required | Number of subclasses |
| `-batch_size` | int | `64` | Batch size |
| `-forget_subclass_name` | str | required | Subclass to forget |

**random_:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-classes` | int | required | Number of classes |
| `-forget_perc` | float | required | Percentage of training set to forget |
| `-batch_size` | int | `128` | Batch size (default higher for random strategy) |

### Optional flags

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-force_re_evaluation` | flag | `False` | Re-evaluate even if results exist |
| `-track_evaluation_resources` | flag | `False` | Track time/memory/power for each evaluation metric |
| `-force_reunlearning` | flag | `False` | Re-run unlearning even if artifacts exist |
| `-skip_evaluation_if_logged` | flag | `False` | Skip evaluation if W&B already has results |
| `-cleanup_checkpoints_after_eval` | flag | `False` | Delete checkpoints after evaluation completes |
