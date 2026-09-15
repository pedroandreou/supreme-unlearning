# Running experiments

The pipeline runs **train → unlearn → evaluate** automatically. Re-running is safe: per-stage outputs (training checkpoints, unlearning checkpoints, already-logged W&B results) are detected and skipped.

## Local (workstation, GPU server, interactive cluster node)

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
| `--gpu` | GPU ID(s) - `0` single, `0,1,2,3` multi-GPU (multi-GPU uses DDP by default; override with `DISTRIBUTED_STRATEGY=fsdp\|deepspeed`) | `0` |
| `--models` | `ResNet18`, `ViT` | both |
| `--training-seeds` | Comma-separated training seeds (outer loop, `I`). | `260`–`269` |
| `--unlearning-seeds` | Space-separated indices for `J` (e.g. `"0 1 2"` for `J=3`) | `"0"` (matched) |
| `--evaluation-seeds` | Space-separated indices for `K` | `"0"` (matched) |
| `--methods` | Unlearning methods to run | all 13 (11 methods + Retrain baseline + Original reference) |
| `--strategies` | Unlearning scenarios to run: `fullclass`, `subclass`, `random_` | all |
| `--datasets` | Datasets to use | all 5 |
| `--forget-percs` | Forget % for the `random_` scenario | `0.001`–`0.10` |
| `DISTRIBUTED_STRATEGY` (env var) | Distributed training strategy for multi-GPU runs: `ddp`, `fsdp`, `deepspeed` - unrelated to `--strategies`, which selects unlearning scenarios | `ddp` |

## SLURM (HPC, login node)

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

# 4 GPUs per job; multi-GPU jobs use DDP by default
# (override with DISTRIBUTED_STRATEGY=fsdp|deepspeed)
./src/supreme/run_slurm.sh --gpus 4
```

Each submitted job runs one `(seed, dataset, model)` cell independently; cells run in parallel across the cluster. Distributed-strategy selection (DDP / FSDP / DeepSpeed) is documented in [`docs/implementation_notes.md → Distributed Strategies`](implementation_notes.md#distributed-strategies).
