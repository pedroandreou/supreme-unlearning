# Implementation Notes (for contributors & paper reviewers)

## Inter-Process Communication & Coordination

Lightning Fabric provides [four core distributed operations](https://lightning.ai/docs/fabric/2.1.0/advanced/distributed_communication.html) for coordinating multi-GPU training: **Barrier**, **Broadcast**, **Gather**, and **Reduce**.

**Setup Phase (rank 0 → all ranks):**
- Main process (rank 0) loads datasets and initializes models
- Uses `fabric.barrier()` and `fabric.broadcast()` to distribute to all ranks
- Prevents redundant I/O and ensures identical starting conditions

**Result Aggregation (all ranks → combined result):**
- **Our choice: `fabric.all_gather()` over `fabric.all_reduce()`**
- **Rationale:** `all_gather()` collects data from each GPU without modification, preserving per-GPU values. This allows manual reduction operations post-gathering, enabling inspection of individual GPU values for debugging and validation
- We verified that the number of GPUs does not affect final results

**Common aggregation patterns:**
- `.mean()`: Average across processes (e.g., accuracy, loss)
- `.max()`: Maximum across processes (e.g., execution time, peak memory)
- `.sum()`: Sum across processes (e.g., total correct predictions)

**Reference:** [Lightning Fabric Distributed Communication](https://lightning.ai/docs/fabric/2.1.0/advanced/distributed_communication.html)

---

## Distributed Gradient Operations

When adapting existing unlearning methods for distributed computing, we Standardised gradient computation across all methods for consistency and correctness:

- **Unified approach:** All unlearning methods-training-based (Bad Teacher, Random Labels/Amnesiac, Finetune, UNSIR) and tuning-based (SSD, LFSSD)-use `fabric.backward()` for gradient computation. This ensures proper gradient handling across all precision modes (fp32, fp16, bf16) and maintains consistency throughout the codebase.

- **Methods with parameter updates:** Training-based methods and UNSIR's noise generation call `optimizer.step()` after `fabric.backward()`, utilizing the synchronized gradients for distributed parameter updates.

- **Methods without parameter updates:** Tuning-based methods (SSD, LFSSD) use `fabric.backward()` for gradient computation but never call `optimizer.step()`. The gradients are squared (SSD) or taken as absolute values (LFSSD) for importance scores, then aggregated across GPUs using `fabric.all_reduce()`. While the gradient synchronization from `fabric.backward()` is technically redundant for these methods, we prioritize correctness and consistency over the minor performance optimization.

**Rationale:** This unified approach ensures:
1. Correct gradient scaling/unscaling in all precision modes
2. Consistent behavior across all unlearning methods
3. Future compatibility with Lightning Fabric updates
4. Simplified codebase maintenance

**Implementation Note:** The SSD/LFSSD implementations create and setup optimizers that are only used for `zero_grad()` calls, never for optimization. This pattern is preserved from the original implementations for reference consistency.

---

## SyncBatchNorm for Multi-GPU Training

Standard BatchNorm computes statistics only from the local batch on each GPU. **SyncBatchNorm** synchronizes these statistics across all GPUs, ensuring consistent normalization.

The framework automatically converts all models to SyncBatchNorm in [unlearn_main.py](../supreme/utils/unlearning/unlearn_main.py#L425-L430) before `fabric.setup()`. This conversion must happen before Fabric wraps the model.

**Implementation:** [convert_to_sync_batchnorm()](../supreme/utils/fabric/fabric_setup.py#L257-L272)

---

## Multi-GPU Batch Size Scaling Strategy

In DDP (Distributed Data Parallel), each GPU processes its own batch independently, then gradients are synchronized. This means:

```
effective_batch_size = per_gpu_batch_size × num_gpus
```

**Problem:** If you use `batch_size=64` on 4 GPUs, your effective batch size becomes 256, not 64. This changes training dynamics and produces different results than single-GPU training.

**Our Solution:** We automatically scale `batch_size` down by `num_gpus` to maintain the same effective batch size. This is handled centrally in [`create_dataloader()`](../supreme/utils/generic_utils.py#L56-L93):

```python
if num_gpus > 1:
    scaled_batch_size = batch_size // num_gpus
    if scaled_batch_size < 1:
        scaled_batch_size = 1
    batch_size = scaled_batch_size
```

**Example:** With `batch_size=64` on 4 GPUs:
- Per-GPU batch size: 64 ÷ 4 = 16
- Effective batch size: 16 × 4 = 64 (same as single-GPU)

**Why NOT learning rate scaling?**

Some frameworks scale learning rate linearly with batch size (`lr = base_lr × num_gpus`). We chose NOT to do this because:
1. **Simplicity:** Batch size scaling alone achieves result consistency
2. **Method compatibility:** Some unlearning methods have carefully tuned learning rates that shouldn't change
3. **Reproducibility:** Matching single-GPU results exactly is our priority
4. **Research validity:** Changing LR could affect unlearning method comparisons

**Verification:** With batch-size scaling, 4-GPU runs match single-GPU runs to within ~1% on most metrics; disabling the scaling produces 5-10% drift.

**Other scaling strategies.** Batch-size scaling is one of several options - linear LR scaling, gradient accumulation, square-root LR scaling, and warmup variants all trade off reproducibility, throughput, and final accuracy differently. For a broader walkthrough of multi-GPU training trade-offs in the deep-learning setting, see Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository. Users who care more about throughput than exact-match reproducibility with single-GPU baselines may prefer one of those alternatives.

---

## Multi-GPU Scaling Performance Guidelines

Not all experiments benefit equally from multi-GPU distributed training. Based on empirical analysis comparing single-GPU and 4-GPU configurations:

**When to Use Multi-GPU (4+ GPUs):**

| Criterion | Recommendation |
|-----------|----------------|
| **Single-GPU time > 60 seconds** | Scale to multi-GPU |
| **Single-GPU time 30-60 seconds** | Test both configurations |
| **Single-GPU time < 30 seconds** | Keep single-GPU |

**Model Size Guidelines:**

| Model Size | Parameters | Multi-GPU Benefit | Expected Speedup |
|------------|------------|-------------------|------------------|
| **Large** (e.g., ViT) | >50M | Always beneficial | ~3x on 4 GPUs |
| **Small** (e.g., ResNet18) | <20M | Often slower | 0.5-1.0x (communication overhead dominates) |

**Why Small Models Don't Scale Well:** The communication overhead in DDP (gradient synchronization, all-reduce operations, PCIe/NVLink bandwidth) dominates when computation time is already minimal.

---

## UNSIR under Fabric/DDP: Noise Module Call Convention

To make the UNSIR noise training compatible with Fabric/DDP wrapping, we ensure the tiny noise module receives a non-empty tensor input:

- **Problem:** Calling a wrapped module with no inputs (e.g., `noise()`) leads to an empty input tuple, which triggers an IndexError inside distributed pre-forward hooks.
- **Solution:** Define `UNSIR_noise.forward(dummy: Tensor = None)` and call it with a 1-element dummy tensor on the correct device (we ignore the value). This preserves DDP semantics while keeping the learnable noise tensor as the actual output used by the model.
- **Where:** See `supreme/methods/unlearning_methods/unsir.py` (`UNSIR_noise.forward` and the call site in `UNSIR_noise_train`).
- **Alternatives:** You could avoid wrapping the noise module with Fabric and manually synchronize parameters/gradients across ranks, but the dummy-input approach is simpler and keeps the unified `fabric.backward(loss)` design intact.

---

## Vision Transformer (ViT) Configuration

For Vision Transformer models, the framework handles specific configuration requirements automatically:

- **Training Phase:** The `find_unused_parameters` flag is enabled to properly handle pooler layer weights that are newly initialized when adapting HuggingFace pretrained models for our specific datasets and tasks.

- **Unlearning Phase:** The `find_unused_parameters` flag is also enabled when using unlearning methods with incompetent teachers (e.g., randomly initialized weights in the Bad Teacher framework), while other unlearning methods use trained model checkpoints as starting points.

This configuration ensures proper gradient computation and parameter updating across all distributed training scenarios with ViT models.

---

## Memory Management

Generic cleanup is strategically invoked both within experiments and at their conclusion.

- **At experiment conclusion**: We perform cleanup following guidance from the optimization literature, including (i) GPU→CPU tensor migration, (ii) explicit variable dereferencing, (iii) forced garbage collection, and (iv) GPU cache clearing.

---

## Non-Adopted Optimizations & Rationale

We evaluated several optimizations but did not adopt them in the final experiments. We document them here for completeness and to aid practitioners who may face similar constraints.

- **In-place weight assignment** (`assign=True`, introduced in PyTorch 2.3.0) was *not* adopted. The codebase depends on TorchAudio 2.1.0, which is only compatible with PyTorch 2.1.0 and therefore lacks support for newer PyTorch versions that provide `assign=True`.

- **DataLoader drop_last**: We retained `drop_last=False`. Although some optimization guides recommend `drop_last=True`, enabling it caused uneven batch distribution across processes under DDP when dataset sizes were not perfectly divisible, leading to synchronization failures and potential deadlocks in collective operations.

- **Compilation and fused kernels** were avoided despite potential performance gains:
  - `torch.compile()` led to symbolic size/stride errors for ViT due to dynamic shape handling, and `fabric.setup()` wraps models in a way that triggers recompilation, negating pre-compilation benefits. Additionally, `deepcopy()` doesn't preserve compilation status, requiring recompilation when copying models (e.g., for retrained model comparisons).
  - Fused CUDA kernels are primarily optimized for `AdamW`, whereas the vision unlearning literature and our experiments use `Adam`.

  **For researchers wanting to enable `torch.compile()`:** The main challenges are (1) handling dynamic shapes in ViT models, and (2) managing recompilation after `fabric.setup()` and `deepcopy()` operations. Potential solutions involve configuring `torch._dynamo.config.automatic_dynamic_shapes = True` and exploring symbolic shape handling. Key resources: [[1]](https://discuss.pytorch.org/t/cannot-call-sizes-on-tensor-with-symbolic-sizes-strides/187807), [[2]](https://github.com/pytorch/pytorch/issues/99774), [[3]](https://github.com/pytorch/pytorch/issues/103892), [[4]](https://github.com/pytorch/pytorch/issues/96414), [[5]](https://github.com/mlcommons/algorithmic-efficiency/issues/496), [[6]](https://github.com/pytorch/pytorch/issues/122171), [[7]](https://github.com/pyg-team/pytorch_geometric/issues/8747). See also: [PyTorch dynamic shapes documentation](https://pytorch.org/TensorRT/user_guide/dynamic_shapes.html).

---

## Data Distribution Export & Auditability

SUPREME exports class distributions and sample-level details for each forget and retain set, making data splits fully auditable across experiments. This functionality is controlled by the `export_class_distribution_info_flag` parameter:

- **Automatic Export:** When enabled, the framework exports detailed CSV files containing class distributions and sample-level information for all training and test sets, as well as retain and forget subsets.

- **Full Auditability:** Each exported file includes batch indices, sample indices, class names, and counts, enabling researchers to verify data splits, debug unexpected results, and ensure reproducibility.

- **Implementation:** The core export functionality is implemented in `debug_utils.py` with integration points in both `training_utils.py` and `unlearning_utils.py` to capture distributions at different stages of the pipeline.

---

## Unlearning Method DataLoader Implementation

The upstream reference implementations of Bad Teacher, Random Labels, and UNSIR materialised the entire training set into a Python list before any training step began. On ViT experiments this consumed ~30 GB of CPU memory and added several hours of wall-clock time with zero GPU utilisation, inflating the measured `time_elapsed`, `TotalCPUMemoryGB`, and `TotalAverageComputeUtil` metrics without affecting model weights.

**Fix:** We replaced the list-building loops with `torch.utils.data.Dataset` wrappers (`RandomLabelDataset`, `NoisyRetainDataset`, `RetainRelabelDataset`, `torch.utils.data.Subset`) that load samples on-the-fly via `__getitem__`. This reduces peak CPU memory to ~1 GB and brings runtimes down to under 10 minutes.

**Result equivalence:** The unlearning trainset is constructed with `unlearning=True`, which selects deterministic transforms (resize + centre crop + normalise only, no random augmentation). Because `__getitem__` always returns the same tensor for the same index, the Dataset wrapper and the materialised list produce bitwise-identical training data. All evaluation metrics (accuracy, MIA, activation distance, completeness, JS-divergence, layerwise distance) are unaffected.

**Affected W&B runs:** Experiments completed before this fix have correct accuracy/MIA results but inflated time and CPU memory metrics. Those runs must be rerun to obtain valid efficiency measurements.

---

## Known Limitations

**GPU Resource Monitoring:**
- On NVIDIA: uses NVML for SM utilization (nvidia-smi as fallback)
- On Apple Silicon (MPS): uses `ioreg` Renderer Utilization % (closest equivalent to SM utilization for compute workloads)
- Tracks compute core utilization rather than total GPU power
- **Rationale:** Compute utilization is process-specific and accurate in multi-user environments, whereas total board power reflects all processes and cannot be attributed to individual workloads

**Apple Silicon (MPS) Limitations:**
- Single-device only - no multi-GPU DDP support (strategy is set to `"auto"`)
- BFloat16 (`bf16-mixed`) not supported in PyTorch 2.1.0 on MPS; use `32-true` or `16-mixed`
- BitsandBytes quantization (nf4, fp4, int8) is CUDA-only and raises an error on MPS
- No hardware peak memory counter - peak GPU memory is tracked via a background polling thread (~100ms interval)
- Significantly slower than NVIDIA A100/L40S (~20-30x for ViT training) due to hardware differences; suitable for development and debugging, not full-scale experiments

**DDP Scalability:**
- Data-parallel metrics achieve near-linear speedup with additional GPUs
- Parameter-based metrics (layerwise distance) require manual parameter sharding across ranks (see [metrics_main.py:356-379](../supreme/eval_metrics/metrics_main.py#L356-L379))
- Time metric remains unaffected (simple arithmetic on pre-computed values)

**Reproducibility:**
- Results not bit-exact between single-GPU and multi-GPU settings due to non-deterministic PyTorch operations (e.g., ResNet's `AdaptiveAvgPool2d`)
- Observed differences remain consistently small (typically <1%) and don't affect experimental conclusions

---

## Distributed Strategies

SUPREME supports three distributed strategies via the `DISTRIBUTED_STRATEGY` environment variable or the `-distributed_strategy` CLI argument:

| Strategy | Description | Best for | Requirements |
|----------|-------------|----------|--------------|
| [**DDP**](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.DDPStrategy.html) | Data-parallel: replicates the full model on each GPU, synchronises gradients | Default choice, small-to-medium models | PyTorch (built-in) |
| [**FSDP**](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.FSDPStrategy.html) | Fully-sharded: shards model parameters across GPUs, gathers on demand | Large models that don't fit on a single GPU | PyTorch ≥ 1.12 (built-in) |
| [**DeepSpeed**](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.DeepSpeedStrategy.html) | ZeRO: shards optimiser / gradient / parameter state across GPUs (stages 1-3) | Memory-efficient training with fast throughput | `pip install deepspeed==0.14.5` |

**DeepSpeed [ZeRO stages](https://www.deepspeed.ai/tutorials/zero/)** (set via `DEEPSPEED_STAGE` env var or `-deepspeed_stage` CLI arg, default: 2):

| Stage | What's sharded | Memory savings | Communication overhead | Notes |
|-------|---------------|----------------|-----------------------|-------|
| **1** | Optimiser states | Moderate | Low | Least intrusive; good starting point |
| **2** | Optimiser states + gradients | Good | Moderate | Best balance for most workloads (default) |
| **3** | Optimiser states + gradients + parameters | Maximum | Higher | Uses the [Infinity Engine](https://www.deepspeed.ai/tutorials/zero-offload/); supports CPU/NVMe offloading via `offload_parameters` / `offload_optimizer` |

```bash
# Local multi-GPU with different strategies
DISTRIBUTED_STRATEGY=ddp ./supreme/run_local.sh --gpu 0,1 --datasets Cifar100 --models ResNet18
DISTRIBUTED_STRATEGY=fsdp ./supreme/run_local.sh --gpu 0,1,2,3 --datasets Cifar100 --models ResNet18
DISTRIBUTED_STRATEGY=deepspeed ./supreme/run_local.sh --gpu 0,1 --datasets Cifar100 --models ResNet18

# DeepSpeed with ZeRO Stage 3 (maximum memory savings)
DISTRIBUTED_STRATEGY=deepspeed DEEPSPEED_STAGE=3 ./supreme/run_local.sh --gpu 0,1,2,3 --models ResNet18

# SLURM with FSDP
DISTRIBUTED_STRATEGY=fsdp ./supreme/run_slurm.sh --gpus 4
```

Checkpoint paths automatically include the strategy (e.g. `2gpus/dist_ddp/`, `4gpus/dist_fsdp/`, `2gpus/dist_deepspeed_stage2/`) so different strategy runs never collide. Single-GPU runs use `no_dist/` instead.

> **Note.** FSDP and DeepSpeed require multiple GPUs. If selected with a single GPU, a warning is printed and the strategy still runs but provides no benefit. DDP is the default and recommended strategy for most workloads.

### Choosing a strategy

**ResNet18 (~11M parameters) - small model:**

| Priority | Recommended | Why |
|----------|-------------|-----|
| Speed | DeepSpeed Stage 1 or 2 | Overlapped gradient communication and reduce-scatter make it faster than DDP. Stage 1 has the least overhead. |
| Simplicity | DDP | No extra dependencies. Good enough for most cases. |
| Avoid | FSDP / DeepSpeed Stage 3 | Parameter sharding adds all-gather overhead on every forward/backward pass. For a small model that fits in GPU memory this overhead is pure cost. |

**ViT (~86M parameters, HuggingFace `google/vit-base-patch16-224`) - medium model:**

| Priority | Recommended | Why |
|----------|-------------|-----|
| Speed | DeepSpeed Stage 2 | Best balance of communication overlap and memory savings. Gradient sharding reduces AdamW memory pressure. |
| Memory-constrained | FSDP (`FULL_SHARD`) or DeepSpeed Stage 3 | Shards parameters across GPUs, allowing larger batch sizes or smaller-VRAM GPUs. |
| Multi-node | FSDP (`HYBRID_SHARD`) | Shards within a node (fast NVLink), replicates across nodes (slower interconnect). Not yet exposed as a CLI option. |
| Simplicity | DDP | Still works well for ViT on modern GPUs (A100 40/80GB). |

**Rules of thumb:**

- Model fits comfortably in GPU memory → DDP or DeepSpeed Stage 1/2 for speed
- Running out of GPU memory → FSDP or DeepSpeed Stage 3 to shard parameters
- Maximum throughput, memory not a concern → DeepSpeed Stage 1 (least communication)
- **FSDP vs DeepSpeed Stage 3** both shard parameters, but DeepSpeed has more tuning knobs (CPU/NVMe offloading, activation checkpointing). FSDP is simpler and built into PyTorch with no extra dependencies.

### How inference-only models are wrapped during unlearning

The unlearning pipeline sets up multiple models per run: one trainable model (the one being unlearned) and several inference-only models (the original frozen model, the retrained reference, and optionally an unlearning teacher). SUPREME handles these asymmetrically depending on the distributed strategy:

| Strategy | Trainable model | Inference-only models (original, retrained, teacher) |
|----------|-----------------|------------------------------------------------------|
| **DDP** | `fabric.setup(model, optimizer)` wraps with DistributedDataParallel | `fabric.setup(model)` wraps with DistributedDataParallel |
| **FSDP** | Wrapped via `fabric.setup(model, optimizer)` inside each unlearning method (after `reset_parameters()` or parameter surgery runs on the raw model) | `fabric.setup_module(model)` - wraps with FullyShardedDataParallel and **actually shards parameters** across GPUs for a real memory benefit over DDP |
| **DeepSpeed** | Single DeepSpeed engine created via `fabric.setup(model, optimizer)` inside each unlearning method | `model.to(device)` only - **not wrapped, replicated on each GPU** (same memory footprint as DDP) |

**Why the DeepSpeed asymmetry?** Two unrelated upstream limitations combine to prevent proper DeepSpeed sharding of inference-only models:

1. **DeepSpeed ZeRO Stage 1/2 does not support `optimizer=None`** at engine initialization. See the open upstream issue [deepspeedai/DeepSpeed#1699](https://github.com/deepspeedai/DeepSpeed/issues/1699): *"being able to initialize stages 1, 2 engine w/ optimizer=None"*. Only Stage 3 supports it.

2. **Multiple DeepSpeed engines crash `fabric.backward()`** even when passing `model=model` explicitly. See the open Lightning issue [Lightning-AI/pytorch-lightning#19773](https://github.com/Lightning-AI/pytorch-lightning/issues/19773). Calling `backward()` on the second of multiple DeepSpeed engines fails with `TypeError: 'NoneType' object is not subscriptable`.

The combined effect: inference-only models can't be wrapped as DeepSpeed engines (limitation #1 for Stages 1/2) and multiple DeepSpeed engines can't be created at all (limitation #2). So for DeepSpeed, we wrap only the trainable model and leave inference-only models replicated on each GPU. This means DeepSpeed provides memory benefits **only for the model being trained/unlearned**, not for the reference models. For ResNet18 / ViT this is fine because the inference models fit easily; for much larger reference models (e.g. multi-billion-parameter LLMs), switch to FSDP - it properly shards inference models via `fabric.setup_module()`.

**Why the FSDP trainable model is wrapped inside the unlearning method, not at setup time.** Several unlearning methods modify parameters directly (e.g. retrain calls `layer.reset_parameters()`). On an FSDP-wrapped model, parameters are flattened into a `FlatParameter` and direct modification does not persist reliably - see the closed-as-won't-fix PyTorch issue [pytorch/pytorch#107081](https://github.com/pytorch/pytorch/issues/107081). The PyTorch team recommends FSDP2, which requires PyTorch 2.4+ (we use 2.1). To avoid silent-correctness bugs, the pipeline passes the raw trainable model to each method, lets the method run its parameter surgery, then wraps via `fabric.setup(model, optimizer)` at the start of the training loop.

**Parameter-surgery methods fall back to replicated mode under FSDP and DeepSpeed.** Methods that compute gradient-based importance scores and modify weights in-place (SSD, LFSSD, ASSD) are **not wrapped** under FSDP or DeepSpeed. They skip `fabric.setup()` for non-DDP strategies and run on the raw replicated model:

- **FSDP** - Implicit all-gather during forward and reduce-scatter during backward desynchronise ranks when combined with per-parameter iteration patterns, causing NCCL deadlocks. `summon_full_params` cannot be used during forward/backward passes (per the [PyTorch docs](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params)).
- **DeepSpeed** - In-place modifications via `modify_weight()` do not persist correctly on DeepSpeed-wrapped models. The weight dampening silently fails (verified empirically: forget_acc ≈ 80-97 % instead of ≈ 0 % when wrapped, vs correct 0 % when unwrapped).

The computation is still correct: importances are computed locally with `loss.backward()`, then aggregated across ranks via `fabric.all_reduce()`. Weight modifications apply identically on all ranks since the importance tensors are synchronised. However, these methods **do not benefit from FSDP/DeepSpeed memory sharding** - the model is fully replicated on each GPU, same as DDP. Standard-training methods (retrain, finetune, bad_teacher, random_labeling, scrub, etc.) are fully wrapped and benefit from the chosen strategy.

**Checkpoint format by strategy.** Training and unlearning checkpoints are saved via `fabric.save()` for DDP and FSDP, which handles strategy-specific unwrapping automatically and produces a `{"model": state_dict}` format. For DeepSpeed, `fabric.save()` produces a sharded directory incompatible with `torch.load()`, so we extract the raw model's state_dict and save it directly via `torch.save()`. Both formats are loaded transparently by `load_weights_efficiently()`. DeepSpeed checkpoints do not include optimiser states - only model weights - but training and unlearning always run to completion (no mid-epoch resumption), so optimiser state isn't needed.

**Correctness note for `layerwise_distance`.** FSDP inference models have sharded parameters; iterating `named_parameters()` and doing element-wise comparison on local shards would give wrong results. [`metrics_main.py`](../supreme/eval_metrics/metrics_main.py) detects FSDP-wrapped models and wraps the `lay_dist` call in `FSDP.summon_full_params(..., writeback=False)` for both models, temporarily gathering full params on every rank. Brief memory spike, but guaranteed correctness. Other metrics (accuracy, jsdiv, activation_distance, completeness, time, ZRF, MIA) use only forward passes and are unaffected.
