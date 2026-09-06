# Distributed Stage 3 evaluation

For the subsequent PyTorch 2.1.0 / Fabric 2.1.0 / DeepSpeed 0.14.5 testing,
including ViT and seven-GPU runs, see [the pinned-stack validation record](stage3_pinned_validation.md).
The dated records below describe what was established at each earlier point.

Stage 3 is a fresh, inference-only worker group. It loads existing checkpoints
before wrapping models, then distributes evaluation data and metric computation.
It does not reuse the Stage 2 optimizer or process group, and it fails if the
required unlearned checkpoint/logs are missing instead of silently training.

## Model layout

| Selected strategy | Evaluation parameters | Implementation |
| --- | --- | --- |
| DDP | Replicated per device | Fabric/DDP forward wrapper |
| FSDP | Sharded | Fabric/FSDP wrapper for every participating model |
| DeepSpeed ZeRO 1 | Replicated per device | Optimizer-free precision wrapper in the DeepSpeed process group |
| DeepSpeed ZeRO 2 | Replicated per device | Same as ZeRO 1 during inference |
| DeepSpeed ZeRO 3 | Sharded | Separate DeepSpeed inference engine for each model |

ZeRO 1 shards optimizer state; ZeRO 2 additionally shards gradients. Neither
state exists in Stage 3. Replication is therefore the intended inference layout,
not an unannounced fallback from parameter sharding. ZeRO 3 additionally shards
parameters, including during inference. See the [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html).

DeepSpeed 0.15 can allocate a BF16 optimizer wrapper and gradient buffers even
when initialized with `optimizer=None`. The ZeRO 1/2 evaluation wrapper avoids
those unused allocations, preserves DeepSpeed's forward dtype conversion, and
synchronizes parameters and buffers once at setup. All metric communication and
data sampling still use the selected Fabric/DeepSpeed process group.

For FSDP evaluation, parameters, inputs and buffers use a consistent compute
dtype. BatchNorm's default FSDP fp32 exception is disabled during inference to
avoid incompatible fp32 activations and bf16/fp16 weights.

## Data and result aggregation

Every data-dependent metric evaluates a shard of the dataset on each device.
Fabric installs a `DistributedSampler`. Its padding keeps forward-call counts
equal, which is necessary when model forwards contain collectives. A validity
mask excludes padding from every metric, including when the dataset contains
fewer samples than devices. Dropping the final batch is rejected.
Sampler rank and replica counts come from the launched process group, not the
number of locally visible GPUs. This also covers SLURM's one-visible-GPU-per-task
binding and external launchers whose process count differs from device visibility.

All-gather is a collective: **every rank calls it and every rank receives the
gathered values**. It is not a rank-zero-only gathering operation. Each rank
combines the gathered sufficient statistics to obtain the same global result;
only rank zero writes JSON or logs to W&B. A global mean is calculated from
global sums and counts, not an unweighted mean of unequal local means.

| Metric | Distributed work | Combination |
| --- | --- | --- |
| Accuracy and loss | Forward passes, correct counts, loss sums on each GPU | All-gather sums/counts; calculate global ratios |
| ZRF and JS-divergence | Both model forwards and divergence contributions on each GPU | All-gather contribution sums and element counts |
| Activation distance | Both forwards and squared probability differences on each GPU | All-gather sums/counts; apply final square root |
| Completeness | Both forwards and prediction agreement counts on each GPU | All-gather agreement/sample counts |
| Layerwise distance | Parameter differences partitioned by element count | All-gather squared-distance sums; apply final square root |
| MIA | Batched model inference and entropy features on each GPU | All-gather entropy scalars; rank-zero CPU classifier fit/predict; broadcast final score |
| Time metric | Arithmetic on previously recorded Stage 2 timings | No expensive GPU computation to parallelize |

MIA is partly distributed, not a distributed scikit-learn optimizer. Its
classifier failure status is broadcast before its score, so other ranks do not
wait for a score that will never arrive.

Layerwise distance validates parameter names and shapes. FSDP temporarily
materializes full parameters on all ranks, increasing peak memory. ZeRO 3
materializes one parameter pair at a time. Every rank enters these collective
contexts in the same order, including ranks with no arithmetic for that pair.
Comparing a model with itself does not nest the same FSDP context twice.

The existing SUPREME metric definitions are preserved. In particular, the
historical `js_divergence_elements` uses the class-averaged reverse-KL-to-mixture
expression, rather than redefining it as conventional Jensen-Shannon divergence
in this distribution change. Results should be interpreted using that existing
definition; this implementation change is not a validation of every scientific
metric definition.

## Launching and performance

The pipeline retains separate Stage 2 and Stage 3 launches. For standalone
multi-GPU runs, `MAIN.sh` uses `torchrun` to supervise the workers. Direct
multi-GPU `python unlearn_main.py` evaluation calls also re-execute under
`torchrun`. SLURM launches use `srun --kill-on-bad-exit=1`.

On an arbitrary worker failure, the error handler does not call all-gather or
barrier. The failing rank exits nonzero, and the supervisor terminates its
peers. Normal successful cleanup still synchronizes the ranks. This avoids
mixing error-handler collectives with another rank's forward collectives.

During Stage 3, `-batch_size` defaults to **per-device batch size**, so adding GPUs
can reduce forward iterations. Stage 1/2 default to global batch-size scaling;
both modes are now explicit options, described below. Use `-track_evaluation_resources`, or
`TRACK_EVALUATION_RESOURCES=true` through `MAIN.sh`, to measure the metric runs.
Timing synchronizes CUDA around the measured call. Legacy resource logs with
missing fields remain readable; absent measurements are omitted, not invented.
Re-evaluation replaces newly computed scores atomically and retains unrequested
metrics.

Distribution does not guarantee speedup for every metric or model. FSDP and
ZeRO 3 incur parameter communication, MIA retains a serial CPU classifier, and
small datasets can be dominated by launch and collective overhead. Full
checkpoints are loaded before sharding, so initialization and FSDP layerwise
distance can require more memory than steady-state sharded inference.

## Reproducible validation

The download-free CPU suite includes real two-process Gloo tests for padding,
all built-in numerical metric paths, rank-zero classifier failure propagation,
and element-balanced parameter distance:

```bash
python -m pytest -q
```

The CUDA probe asserts the actual parameter layout, tests BatchNorm and
datasets of one and seven samples, compares metrics against unwrapped-model
outputs, checks model aliasing, and verifies inference after parameter gathering:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m torch.distributed.run \
  --standalone --nproc-per-node=2 scripts/probe_stage3_distributed.py \
  --strategy fsdp --precision bf16-true
```

Select `--strategy ddp`, or `--strategy deepspeed --stage 1`, `2`, or `3`.
Adding `--fail-rank 1` deliberately fails one worker while its peer waits in a
barrier; the supervised job must exit nonzero promptly without leftover workers.

For real, existing checkpoints, set `STAGE3_ARTIFACTS` to the artifact-tree root,
`STAGE3_LOG_SUFFIX` to its relative log directory, `STAGE3_WEIGHT` to the original
model checkpoint, and `CUDA_VISIBLE_DEVICES` to idle devices. Then run:

```bash
bash scripts/validate_stage3_checkpoints.sh -track_evaluation_resources
```

The runner checks the imported repository path, copies artifacts into an isolated
temporary directory without old evaluation JSON, and runs a single-GPU baseline
plus DDP, FSDP and all three ZeRO selections. It does not retrain models. Set
`STAGE3_PYTHON` to the required environment's Python executable. Defaults target
ResNet18/CIFAR-10, random 1% forgetting and seed 260; the script exposes environment
overrides for these values.
When the matrix includes a single-GPU baseline and distributed cases, the
runner also invokes `scripts/compare_stage3_results.py`: score keys must match
and numerical differences must be within `STAGE3_COMPARE_ATOL` (default
`0.00001`, absolute tolerance). Runtime/resource values are excluded. The
comparison rejects missing scores and nonfinite numerical results.

## Validation record: 5 September 2026

All changes were made on `fix/distributed-stage3-evaluation`. CUDA runs used the
an isolated checkout on a shared GPU host, with two A100
SXM4 80 GB devices. The original server worktree and checkpoints were not
modified. Import paths were checked to exclude the unrelated editable install.

| Check | Result |
| --- | --- |
| Local CPU suite, PyTorch 2.1.0 / Lightning 2.1.0 | 132 passed |
| Existing checkpoints, single GPU and two-GPU DDP/FSDP/ZeRO 1/2/3, bf16-true | All built-in metrics completed |
| Same checkpoint matrix with resource tracking and legacy per-process logs | All six cases completed and wrote JSON |
| Two-GPU numerical/layout probes, 32-true and bf16-true | All five strategies passed |
| Two-GPU numerical/layout probes, bf16-mixed | All five strategies passed |
| Rank-one failure while rank zero waits | All five jobs exited nonzero through torchrun; no timeout or surviving probe workers |

The CUDA environment was PyTorch 2.4.1+cu121, Lightning 2.1.0, DeepSpeed 0.15.0,
and scikit-learn 1.6.1. This is not a claim that the pinned PyTorch 2.1 CUDA stack,
multi-node SLURM, every model architecture, or every precision combination has
been GPU-tested.

The existing ResNet18/CIFAR-10 checkpoints used random 1% forgetting, seed 260,
and 10,000 test samples. Whole-test accuracy was 87.99%, forget accuracy 89.0%,
MIA approximately 0.738, and whole-test completeness 89.16%. These values matched
the single-GPU baseline across all five distributed strategies. The maximum
absolute difference across the recorded numerical scores was
`8.249771421553476e-09`. This checks distribution equivalence for these
checkpoints, not the scientific validity of the historical metric definitions.

The 265,219-parameter probe confirmed FSDP local element counts of 132,610 and
132,609, ZeRO 3 local partition storage of 132,610 elements per rank (including
padding), and full 265,219-element replicas for ZeRO 1/2. ZeRO 3 may also persist
small parameters, so its partition counts are not a complete memory estimate.

Private evidence includes the complete checkpoint matrix, resource tracking,
JSON scores, numerical/layout probes and intentional rank-local failures.
An initial mixed-ZeRO oracle assertion was corrected to account for DeepSpeed's
low-precision inference weights. Raw logs and infrastructure-specific paths
are deliberately excluded from this public record.

These were correctness checks on a shared server, not a controlled scaling
benchmark. The measured accuracy pass was 1.27 s on one GPU, 1.14 s with DDP,
1.34 s with FSDP and 1.68 s with ZeRO 3 in one resource-tracked run. This is
evidence against promising universal speedup for a small model, even when its
computation and model storage are correctly distributed.

## Explicit batch policies and larger jobs

There is no fixed GPU-count constant in the Stage 3 metric implementation.
The data-parallel world size is the number of launched workers, normally
`nodes * GPUs per node`, with one worker per GPU. This is a configuration rule,
not a statement that any arbitrary world size has been validated.

`-batch_size_mode global|per_device` controls the meaning of `-batch_size` in
the Python training/unlearning entry points. Without that argument:

- Stage 1/2 use `BATCH_SIZE_MODE`, default `global`.
- Stage 3 uses `EVALUATION_BATCH_SIZE_MODE`, default `per_device`.

For requested batch size 128, full microbatches have these nominal sizes:

| Mode | GPUs | Samples per GPU | Global microbatch |
| --- | ---: | ---: | ---: |
| global | 1 | 128 | 128 |
| global | 8 | 16 | 128 |
| per_device | 8 | 128 | 1,024 |
| per_device | 1,024 | 128 | 131,072 |

The last row is arithmetic, not a hardware benchmark. Final batches can be
smaller; evaluation sampler padding does not count as additional real samples.
Global mode requires exact divisibility by world size. It now raises an error
instead of silently flooring the division or clamping a zero batch to one.
For example, global batch 128 cannot run with 1,000 workers under this policy.
Choose per-device mode or an appropriately divisible global batch instead.
These sizes exclude gradient accumulation; this change does not introduce a
uniform accumulation option across all unlearning methods.

For scalable training/unlearning, set `BATCH_SIZE_MODE=per_device` before the
existing `MAIN.sh` command. Its checkpoint/log paths gain `batch_per_device/`,
and its W&B prefix gains `_batch_per_device`, avoiding accidental reuse of
the default-mode results. Existing valid global-mode commands retain their
paths. Direct Python calls with an explicit `LOG_DIR` must use a distinct log
directory when changing the experiment. Changing batch values within one mode
still requires separate run directories; the namespace is not a complete
hyperparameter cache key. Some methods have their own internal batch sizes;
the policy applies to those loaders too, without replacing their method-specific
hyperparameters with the entry-point batch value.

Batch mode, requested size, per-device size, nominal global microbatch and actual
world size are printed and recorded in experiment configuration. Stage 3 JSON
includes them in `execution_config`. `DATALOADER_NUM_WORKERS` allows controlling
CPU-worker multiplication across ranks instead of always using eight per rank.

The SLURM dispatcher accepts `--nodes`, `--gpus-per-node` (also `--gpus`), and
`--batch-size-mode`. For example, resource calculation can be inspected without
submitting anything:

```bash
bash src/supreme/run_slurm.sh --dry-run --nodes 128 --gpus-per-node 8 \
  --batch-size-mode per_device
```

Actual jobs require the scheduler allocation, matching software and accessible
code/data/checkpoints on every node, and working rendezvous/NCCL networking.
`MAIN.sh` no longer forcibly disables InfiniBand or overrides NCCL's peer-to-peer
topology discovery. Site-supplied `NCCL_*` variables remain available. External
homogeneous multi-node torchrun launches infer node count from `WORLD_SIZE` and
`LOCAL_WORLD_SIZE`; sampler sizing and batch division use the actual global
process group, not the number of locally visible devices.

### Reproducibility and speed

A smaller per-device batch is not inherently incorrect. Fixed-global batching
is useful for preserving optimization conditions while increasing devices.
However, very small local batches can underutilize GPUs, and local BatchNorm
can impose additional constraints during training. Per-device batching keeps
local work constant but increases the global training batch, changing update
frequency and potentially the learned model. Learning rate is not silently
rescaled by this option. Reproducing a published result requires its complete
training configuration, not only its seed.

Evaluation does not update weights, but batch shape, precision and reduction
order can still change floating-point results. In the expanded synthetic
bfloat16 probe, a logit margin of approximately 0.000366 led to one of 37
predictions changing between a full-batch oracle and four-GPU microbatches.
Exact classification assertions now use inputs with separated top logits;
continuous-score comparisons retain numerical tolerances. This avoids
misidentifying a near-tie rounding effect as a sampling/aggregation failure,
without claiming bitwise invariance across GPU counts.

More GPUs do not guarantee lower latency. Forward work per GPU decreases for
a fixed dataset, but launches, communication, padding, storage and serial work
eventually dominate. For example, a 500-sample forget set cannot give 1,000
ranks distinct useful samples in one evaluation pass. Prefer throughput
benchmarks on representative large workloads, including end-to-end startup,
rather than extrapolating from GPU count or a tiny correctness probe.

### What remains before a thousand-GPU scaling claim

This release is not validated for thousand-GPU production runs. In particular:

- Dataset setup still broadcasts pickled dataset objects from rank zero.
  Large jobs need rank/node-local construction and compact split metadata,
  backed by storage capable of serving the aggregate readers.
- Existing checkpoint loading materializes full models before sharding, and
  FSDP layerwise distance temporarily materializes full parameters on each rank.
  Large-model support needs distributed checkpoint loading and shard-native
  parameter-distance arithmetic.
- MIA gathers entropy arrays to every rank and fits its classifier on rank-zero
  CPU. Feature collection/storage and attack fitting need a separate scaling
  design for large evaluation sets.
- Multi-node network behavior, CPU/I/O contention, failure recovery and
  end-to-end efficiency require tests on an allocated cluster. A launcher
  dry-run or simulated 1,024-rank sampler test cannot establish these properties.

These are remaining engineering and validation requirements, not settings that
can be solved simply by selecting FSDP or ZeRO 3.

## Additional scaling validation: 5 September 2026

The second round used four A100 GPUs on the same shared host and isolated checkout,
without retraining or modifying the source checkpoints:

| Check | Result |
| --- | --- |
| Final local suite, PyTorch 2.1.0 / Lightning 2.1.0 | 167 passed |
| Real CPU Gloo processes at world sizes 2, 3, 4 and 8 | Passed padding, aggregation, metric and classifier-failure checks |
| Four-GPU bf16-true probes, DDP/FSDP/ZeRO 1/2/3, PyTorch 2.4.1 | All passed; 1 and 37 samples; batch 4 per GPU |
| Existing ResNet18/CIFAR-10 checkpoint matrix, one GPU vs four GPUs | All six cases completed; 20 numerical scores agreed within 0.00001 |
| Four-GPU DDP/FSDP probes, PyTorch 2.1.0+cu121 / Lightning 2.1.0 | Passed; global batch 8 resolved to batch 2 per GPU |
| Simulated 1,024-rank samplers, 1/500/1,025 samples | Every real sample counted exactly once |
| SLURM dry-run, 128 nodes times 8 GPUs | Correct 1,024-worker resource calculation; no job submitted |
| Two-node torchrun across physical hosts | Not validated: rendezvous TCP connection timed out before workers started |

The four-GPU checkpoint matrix recorded per-device batch 128 and nominal global
microbatch 512, compared with 128 on the baseline. The maximum absolute score
difference was `2.56416277256e-06`, in loss; reported accuracy, completeness and
MIA remained unchanged. All five distributed strategies agreed with one another.
The initial strict comparison at `1e-6` rejected this loss difference; the
documented tolerance is `1e-5`. The synthetic bfloat16 near-tie finding above
explains why precision/batch-shape effects must be distinguished from incorrect
data partitioning or aggregation.

FSDP stored 66,305/66,305/66,305/66,304 parameter elements per rank in the
265,219-element probe. ZeRO 3 stored 66,305 partition elements per rank, including
padding. ZeRO 1/2 retained full replicas without optimizer/gradient allocations.

Private evidence includes the final four-GPU probes, pinned-stack global-batch
checks, full checkpoint matrix, configurations, resource logs and JSON scores.
The investigated near-tie assertion is retained separately as a failed probe.

Both servers accepted SSH, but peer rendezvous TCP connectivity failed.
The attempted jobs were terminated without changing host/network settings.
Neither server exposed the SLURM submission commands in this session. Therefore
these results establish up to four-GPU Stage 3 correctness for the tested
configurations, not multi-node SLURM execution, thousand-GPU efficiency, or a
new validation of every Stage 1/2 training method.

### Fabric version and API audit

Lightning Fabric and PyTorch have independent version numbers. All test
environments above used **Lightning Fabric 2.1.0**. The complete CUDA strategy
matrix used **PyTorch 2.4.1**, while the additional pinned-stack CUDA probes
used **PyTorch 2.1.0** for DDP/FSDP only. ZeRO 1/2/3 have not been CUDA-validated
on the PyTorch 2.1.0 stack in these runs.

The implementation was checked against the installed Fabric 2.1.0 source and
the official `2.1.0` tag, not inferred from the current stable documentation:

- [Fabric 2.1.0 API source](https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.0/src/lightning/fabric/fabric.py): `setup_module` supports inference; `setup_dataloaders` installs distributed sampling; `all_gather` requires participation and equal tensor shapes across ranks. Variable-row features are padded before gathering.
- [Fabric 2.1.0 DeepSpeed strategy](https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.0/src/lightning/fabric/strategies/deepspeed.py): `setup_module` creates an inference engine without optimizers. ZeRO 3 uses this path.
- [Fabric 2.1.0 FSDP guide](https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.0/docs/source-fabric/advanced/model_parallel/fsdp.rst): wrapping policy, sharding and checkpoint-format tradeoffs. The current full-checkpoint compatibility path is not the guide's preferred large-model sharded-loading workflow.
- [Fabric 2.1.0 SLURM guide](https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.0/docs/source-fabric/guide/multi_node/slurm.rst): scheduler-launched workers and node/device configuration.

The ZeRO 1/2 optimizer-free wrapper, padded-sample masking, actual-world sampler
overrides and one-visible-GPU SLURM environment are SUPREME compatibility code,
not features claimed to be provided unchanged by Fabric. Their tested behavior
and remaining multi-node validation gap are documented above.
