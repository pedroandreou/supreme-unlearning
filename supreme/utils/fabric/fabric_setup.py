import os
import functools
from typing import Any, Dict
import torch
from lightning.fabric import Fabric
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.plugins.environments import LightningEnvironment, SLURMEnvironment
from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.accelerators.cuda import is_cuda_available
from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.accelerators.xla import XLAAccelerator


class SLURMAwareFabric(Fabric):
    """Custom Fabric that uses global_rank for print() instead of local_rank.

    In our SLURM setup with GPU binding, each task sees only 1 GPU, so local_rank
    is always 0 for all ranks (see FlexibleSLURMEnvironment.local_rank()).
    This breaks the default fabric.print() which checks local_rank == 0.

    This subclass overrides print() to use global_rank instead, ensuring only
    the true rank 0 process prints.
    """

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print only from global rank 0.

        The default Fabric.print() checks local_rank == 0, but in our SLURM setup
        all ranks have local_rank == 0. We use global_rank instead.
        """
        if self.global_rank == 0:
            print(*args, **kwargs)


class SLURMAwareDDPStrategy(DDPStrategy):
    """Custom DDP strategy that uses SLURM's world_size and global_rank for DistributedSampler.

    The default DDPStrategy uses len(parallel_devices) for world_size, which is 1 when
    each SLURM task sees only 1 GPU. This causes DistributedSampler to fail with
    "Invalid rank X, rank should be in the interval [0, 0]".

    This strategy overrides distributed_sampler_kwargs to use the actual SLURM values.
    """

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, Any]:
        """Return correct world_size and rank from SLURM environment."""
        # Get values from SLURM environment (the true distributed world)
        slurm_world_size = int(os.environ.get("SLURM_NTASKS", 1))
        slurm_global_rank = int(os.environ.get("SLURM_PROCID", 0))

        return {
            "num_replicas": slurm_world_size,
            "rank": slurm_global_rank,
        }


class FlexibleSLURMEnvironment(SLURMEnvironment):
    """Custom SLURM environment that allows devices != ntasks_per_node.

    This is needed when we manually bind GPUs using SLURM_LOCALID, resulting in
    each task seeing only 1 GPU (devices=1) even though ntasks_per_node > 1.

    Key overrides:
    - _validate_srun_variables(): Skip validation of SLURM_NTASKS vs SLURM_NTASKS_PER_NODE
    - validate_settings(): Skip strict validation of devices vs ntasks_per_node
    - local_rank(): Return 0 since each process sees only 1 GPU after binding
    """

    def _validate_srun_variables(self) -> None:
        """Skip validation of SLURM srun variables.

        The parent class raises an error if SLURM_NTASKS is set without
        SLURM_NTASKS_PER_NODE. We skip this check to support different
        SLURM cluster configurations.
        """
        pass

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        """Skip strict validation of devices vs ntasks_per_node."""
        # Don't call super().validate_settings() to skip the strict check
        # The parent class would raise an error if devices != ntasks_per_node
        pass

    def local_rank(self) -> int:
        """Return 0 since each process sees only 1 GPU after manual binding.

        With our GPU binding fix (_fix_slurm_gpu_binding in train_main.py/unlearn_main.py),
        each SLURM task has CUDA_VISIBLE_DEVICES set to a single GPU.
        Therefore, the local rank within visible devices is always 0.

        This is necessary because DDPStrategy uses local_rank for device indexing:
        `parallel_devices[local_rank]` - and parallel_devices only has 1 element.

        Note: SLURMAwareFabric overrides print() to use global_rank instead of
        local_rank, so fabric.print() correctly prints only from rank 0.

        The actual SLURM_LOCALID (0,1,2,3) is still used for:
        - GPU binding (picking which physical GPU)
        - Global rank calculation via parent class
        """
        return 0


def get_slurm_node_count():
    """Get the number of nodes from SLURM environment, defaulting to 1 for standalone."""
    # SLURM sets SLURM_NNODES or SLURM_JOB_NUM_NODES
    slurm_nnodes = os.environ.get("SLURM_NNODES") or os.environ.get("SLURM_JOB_NUM_NODES")
    if slurm_nnodes:
        return int(slurm_nnodes)
    return 1  # Standalone mode


def _create_distributed_strategy(config, is_slurm, device):
    """Create the distributed strategy based on config['distributed_strategy'].

    Args:
        config: Configuration dict with 'distributed_strategy' and 'model_name' keys
        is_slurm: Whether running under SLURM
        device: Accelerator string ("cuda", "mps", "tpu", or "cpu")

    Returns:
        Tuple of (strategy_object_or_string, strategy_name_string)
    """
    distributed_strategy = config.get("distributed_strategy", "ddp")
    model_name = config.get("model_name", "")

    # TPU/XLA: DDP/FSDP/DeepSpeed are not applicable; force XLA strategy.
    if device == "tpu":
        if distributed_strategy not in (None, "ddp", "auto", "xla"):
            import warnings
            warnings.warn(
                f"Distributed strategy '{distributed_strategy}' is not supported on TPU. "
                "Overriding with 'xla'.",
                UserWarning,
                stacklevel=2,
            )
        return "xla", "xla"

    if distributed_strategy == "xla":
        return "xla", "xla"

    if distributed_strategy == "auto":
        return "auto", "auto"

    if device == "mps":
        if distributed_strategy != "ddp":
            raise RuntimeError(
                f"Distributed strategy '{distributed_strategy}' is not supported on Apple Silicon MPS. "
                "Only 'ddp' (which falls back to 'auto' on MPS) is supported."
            )
        return "auto", "ddp"

    if device == "cpu":
        if distributed_strategy == "ddp":
            find_unused = model_name == "ViT"
            if is_slurm:
                return SLURMAwareDDPStrategy(find_unused_parameters=find_unused), "ddp"
            else:
                return DDPStrategy(find_unused_parameters=find_unused), "ddp"
        raise RuntimeError(
            f"Distributed strategy '{distributed_strategy}' is not supported on CPU. "
            "Use 'auto' or 'ddp' (gloo backend) instead."
        )

    if distributed_strategy == "ddp":
        find_unused = model_name == "ViT"
        if is_slurm:
            return SLURMAwareDDPStrategy(find_unused_parameters=find_unused), "ddp"
        else:
            return DDPStrategy(find_unused_parameters=find_unused), "ddp"

    elif distributed_strategy == "fsdp":
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        # Size-based policy keeps small modules (like BatchNorm) unwrapped,
        # avoiding FSDP + BatchNorm incompatibility issues.
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=1_000_000,
        )
        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            state_dict_type="full",  # Save full (non-sharded) state dict as a single file
        ), "fsdp"

    elif distributed_strategy == "deepspeed":
        try:
            from lightning.fabric.strategies import DeepSpeedStrategy
        except ImportError:
            raise ImportError(
                "DeepSpeed strategy requires the 'deepspeed' package. "
                "Install with: pip install deepspeed"
            )
        # ZeRO stages: 1 = optimizer sharding, 2 = optimizer+gradient, 3 = full parameter sharding
        deepspeed_stage = config.get("deepspeed_stage", 2)
        return DeepSpeedStrategy(stage=deepspeed_stage), f"deepspeed_stage{deepspeed_stage}"

    else:
        raise ValueError(
            f"Unknown distributed strategy: '{distributed_strategy}'. "
            "Choose from: ddp, fsdp, deepspeed, auto, xla"
        )


def initialize_fabric(config):
    """Initialize and configure Lightning Fabric"""

    # Initialize Fabric
    # See tutorial for Multi-GPU Training at
    # https://magazine.sebastianraschka.com/p/accelerating-pytorch-model-training
    #
    # Accelerator priority (auto-detected unless config["accelerator"] is set):
    #   CUDA → MPS → TPU/XLA → CPU (fallback)
    # `config["accelerator"]` ("cuda" | "mps" | "tpu" | "xla" | "cpu" | "auto")
    # can pin the choice; otherwise the first available is used.
    requested = config.get("accelerator")
    if requested == "cpu":
        device = "cpu"
    elif torch.cuda.is_available() and requested in (None, "cuda", "gpu", "auto"):
        device = "cuda"
    elif MPSAccelerator.is_available() and requested in (None, "mps", "auto"):
        device = "mps"
    elif XLAAccelerator.is_available() and requested in (None, "tpu", "xla", "auto"):
        device = "tpu"
    else:
        device = "cpu"

    # Enable tensor cores for faster matrix multiplications if available (CUDA only)
    capability_msg = None
    if torch.cuda.is_available():
        # `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance.
        # For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

        capability = torch.cuda.get_device_capability()
        if (
            capability[0] >= 7
        ):  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
            torch.set_float32_matmul_precision("high")
            capability_msg = "Tensor cores enabled for faster matrix multiplications"
        else:
            capability_msg = (
                "Tensor cores not supported on this GPU. Using default precision."
            )

    # Side note: DDP optimizations in case we want to use them in the future
    # https://lightning.ai/docs/pytorch/stable/advanced/ddp_optimizations.html

    # When running under SLURM, use our custom SLURMAwareDDPStrategy
    # to fix DistributedSampler world_size/rank issues
    is_slurm = os.environ.get("SLURM_JOB_ID") is not None

    # Create the distributed strategy (ddp, fsdp, deepspeed, auto, or xla)
    fabric_strategy, distributed_strategy_name = _create_distributed_strategy(config, is_slurm, device)

    # Warn if FSDP/DeepSpeed is selected but only 1 GPU is available
    multi_gpu = config.get("num_gpus", 1) > 1 or os.environ.get("SLURM_NTASKS", "1") != "1"
    if not multi_gpu and distributed_strategy_name.startswith(("fsdp", "deepspeed")):
        import warnings
        warnings.warn(
            f"Distributed strategy '{distributed_strategy_name}' is selected but only 1 GPU is available. "
            f"FSDP and DeepSpeed are designed for multi-GPU training. "
            f"The strategy will still work but provides no benefit with a single GPU.",
            UserWarning,
            stacklevel=2,
        )

    # Enable SyncBatchNorm for multi-GPU DDP training to synchronize batch normalization
    # statistics across all GPUs. Only for DDP - FSDP and DeepSpeed handle BN differently:
    # - FSDP: size-based wrap policy keeps small BN layers unwrapped (local BN stats)
    # - DeepSpeed: ZeRO doesn't shard small params, BN stays local
    # See: https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
    use_sync_batchnorm = multi_gpu and distributed_strategy_name == "ddp"

    # Get number of nodes (auto-detected from SLURM or default to 1)
    num_nodes = get_slurm_node_count()

    # Sources for precision:
    # https://lightning.ai/docs/fabric/stable/fundamentals/precision.html
    # https://lightning.ai/docs/fabric/stable/api/fabric_args.html
    # https://lightning.ai/docs/pytorch/stable/common/precision_basic.html
    precision = None
    kwargs = {}
    _bnb_modes = ["nf4", "nf4-dq", "fp4", "fp4-dq", "int8", "int8-training"]
    if config["precision"] in _bnb_modes:
        if device != "cuda":
            raise RuntimeError(
                f"Precision '{config['precision']}' uses BitsandBytes which is CUDA-only and "
                f"not supported on '{device}'. Use '32-true', 'bf16-mixed', or '16-mixed' instead."
            )
        # The BitsandbytesPrecision automatically replaces the torch.nn.Linear layers in your model with their BNB alternatives
        # https://lightning.ai/docs/fabric/stable/plugins/bitsandbytes.html
        precision = BitsandbytesPrecision(mode=config["precision"])
        kwargs["plugins"] = precision
    else:
        precision = config["precision"]
        kwargs["precision"] = precision

    # When running under SLURM with manual GPU binding (devices=1 per task),
    # use FlexibleSLURMEnvironment to skip strict validation
    plugins = kwargs.pop("plugins", None)
    if os.environ.get("SLURM_JOB_ID"):
        # Create list of plugins, including FlexibleSLURMEnvironment
        plugin_list = [FlexibleSLURMEnvironment()]
        if plugins is not None:
            plugin_list.append(plugins) if not isinstance(plugins, list) else plugin_list.extend(plugins)
        plugins = plugin_list

    # Build Fabric-native CSV/TensorBoard loggers from flags in `config`.
    # WandbLogger is appended later in `initialize_wandb` (requires auth).
    from supreme.utils.fabric.loggers_setup import build_fabric_loggers
    fabric_loggers = build_fabric_loggers(config)

    # Use SLURMAwareFabric under SLURM to fix print() behavior
    # (standard Fabric.print() checks local_rank==0, but all our ranks have local_rank==0)
    FabricClass = SLURMAwareFabric if is_slurm else Fabric
    fabric = FabricClass(
        accelerator=device,
        devices=config["num_gpus"],
        num_nodes=num_nodes,
        strategy=fabric_strategy,
        callbacks=config["callbacks"],
        loggers=fabric_loggers if fabric_loggers else None,
        plugins=plugins,
        **kwargs,
    )

    if capability_msg:
        fabric.print(capability_msg)

    # Log SyncBatchNorm status for multi-GPU training
    if use_sync_batchnorm:
        fabric.print("SyncBatchNorm enabled for multi-GPU training")

    # Note: For DDP with SLURM:
    # - SLURM spawns processes via srun (--ntasks-per-node=N)
    # - Each process sees 1 GPU (CUDA_VISIBLE_DEVICES set per task)
    # - Fabric detects SLURM and uses world_size from SLURM_NTASKS
    # - devices=1 per process, but world_size=N for distributed training
    # See: https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html
    #
    # SLURM ↔ Fabric Mapping:
    #   --nodes=N           → num_nodes=N (auto-detected from SLURM_NNODES)
    #   --ntasks-per-node=M → devices=M (but with srun, each task sees 1 GPU, so devices=1)
    #   --gpus-per-node=M   → Total GPUs per node (must equal ntasks-per-node)
    #   world_size = num_nodes * ntasks_per_node
    if fabric.world_size > 1 or num_nodes > 1:
        slurm_info = ""
        if os.environ.get("SLURM_JOB_ID"):
            slurm_ntasks = os.environ.get("SLURM_NTASKS", "?")
            slurm_ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE", "?")
            slurm_gpus_per_node = os.environ.get("SLURM_GPUS_PER_NODE", "?")
            slurm_info = f", SLURM[ntasks={slurm_ntasks}, ntasks_per_node={slurm_ntasks_per_node}, gpus_per_node={slurm_gpus_per_node}]"
        fabric.print(f"Distributed training: distributed_strategy={distributed_strategy_name}, world_size={fabric.world_size}, num_nodes={num_nodes}, devices_per_node={config['num_gpus']}, global_rank={fabric.global_rank}{slurm_info}")

    fabric.launch()

    return fabric, device, fabric_strategy, use_sync_batchnorm, distributed_strategy_name


def setup_model_for_inference(fabric, model, distributed_strategy_name):
    """Set up a model for inference (no training) with fabric.

    - DDP: fabric.setup() wraps the model with DistributedDataParallel.
    - FSDP: fabric.setup_module() wraps with FullyShardedDataParallel. This
      actually shards parameters across GPUs for a real memory benefit over DDP.
      setup_module() is the documented way to set up inference models with FSDP:
      https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.fabric.Fabric.html
      With use_orig_params=True (the default in PyTorch 2.0+, fsdp.py:171),
      named_parameters() returns original names and shapes - required by
      layerwise_distance metric (which wraps the iteration in summon_full_params).
    - DeepSpeed: model.to(device) only. DeepSpeed ZeRO Stage 1/2 do not support
      optimizer=None at engine initialization (see deepspeedai/DeepSpeed#1699),
      and creating multiple DeepSpeed engines causes fabric.backward() crashes
      (see Lightning-AI/pytorch-lightning#19773). As a result, DeepSpeed inference
      models are replicated on each GPU - same as DDP - and only the trainable
      model gets a DeepSpeed engine.

    Args:
        fabric: Lightning Fabric instance
        model: The model to set up
        distributed_strategy_name: Name of the distributed strategy (e.g., 'ddp', 'fsdp', 'deepspeed_stage2')

    Returns:
        The model set up for inference
    """
    if distributed_strategy_name == "ddp":
        model = fabric.setup(model)
    elif distributed_strategy_name == "fsdp":
        # fabric.setup_module() is the documented API for inference-only FSDP.
        # It wraps the model in FullyShardedDataParallel with real parameter sharding.
        model = fabric.setup_module(model)
    else:
        # DeepSpeed: upstream limitations force us to replicate inference models.
        # See deepspeedai/DeepSpeed#1699 and Lightning-AI/pytorch-lightning#19773.
        model = model.to(fabric.device)
    return model


def gather_full_state_dict(model, world_size=None):
    """Return the full (unsharded) model state_dict regardless of distributed strategy.

    For FSDP-wrapped models, calling `.state_dict()` directly returns sharded
    tensors (flattened + split across ranks). Loading such a state_dict into a
    raw PyTorch model fails with shape mismatches like `[1728]` vs `[64, 3, 3, 3]`.

    This function replicates the same gathering technique that Lightning Fabric's
    own `FSDPStrategy.save_checkpoint(state_dict_type="full")` uses internally
    (see Lightning Fabric 2.1.0 `fsdp.py:484-498`), so it produces byte-identical
    results to `fabric.save(path, {"model": model})` - but returns the dict in
    memory so we can use the canonical `torch.save(state_dict, path)` pattern
    that the rest of the codebase expects.

    All ranks must call this function together (it triggers NCCL collectives).

    For DDP, DeepSpeed, and non-distributed models, `.state_dict()` already
    returns full tensors - no special handling needed.

    Args:
        model: The (possibly Fabric/FSDP/DDP-wrapped) model.
        world_size: Distributed world size. If None, inferred from torch.distributed.
            Only matters for the CPU-offload heuristic in PyTorch <= 2.0.

    Returns:
        A full (non-sharded) state_dict on every rank with original parameter
        names and shapes. Can be loaded directly into a raw PyTorch model via
        `model.load_state_dict(...)` then saved via `torch.save(state_dict, path)`.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel

    # Detect FSDP by checking the _forward_module (used for actual forward passes).
    # For FSDP, _forward_module contains FullyShardedDataParallel submodules.
    # For DDP, _forward_module is the DistributedDataParallel wrapper (whose
    # state_dict has a "module." prefix we DON'T want).
    fsdp_container = None
    if hasattr(model, "_forward_module"):
        forward_mod = model._forward_module
        has_fsdp = any(
            isinstance(m, FullyShardedDataParallel) for m in forward_mod.modules()
        )
        if has_fsdp:
            fsdp_container = forward_mod

    if fsdp_container is None:
        # DDP, DeepSpeed, or unwrapped models: use .module (the raw unwrapped
        # model) which returns state_dict keys WITHOUT "module." prefix.
        inner = model.module if hasattr(model, "module") else model
        return inner.state_dict()

    # FSDP path: use the exact same context manager that Lightning Fabric's
    # save_checkpoint uses. This was verified to produce correct full tensors
    # via a side-by-side test (fabric.save() vs this helper).
    # CRITICAL: we use _forward_module (fsdp_container) here, NOT .module,
    # because .module returns the raw model bypassing FSDP, whose state_dict
    # would return sharded/flattened tensors.
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp.api import FullOptimStateDictConfig

    if world_size is None:
        try:
            import torch.distributed as dist
            world_size = dist.get_world_size() if dist.is_initialized() else 1
        except Exception:
            world_size = 1

    # Mirrors Lightning Fabric 2.1.0 fsdp.py `_get_full_state_dict_context`:
    offload_to_cpu = world_size > 1
    state_dict_config = FullStateDictConfig(
        offload_to_cpu=offload_to_cpu, rank0_only=False
    )
    optim_state_dict_config = FullOptimStateDictConfig(
        offload_to_cpu=offload_to_cpu, rank0_only=False
    )

    with FullyShardedDataParallel.state_dict_type(
        module=fsdp_container,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=state_dict_config,
        optim_state_dict_config=optim_state_dict_config,
    ):
        state_dict = fsdp_container.state_dict()

    return state_dict


def convert_to_sync_batchnorm(model, use_sync_batchnorm: bool):
    """Convert BatchNorm layers to SyncBatchNorm for multi-GPU training.

    In Fabric (unlike PyTorch Lightning Trainer), SyncBatchNorm conversion
    must be done manually before fabric.setup(model).

    Args:
        model: The model to convert
        use_sync_batchnorm: Whether to perform the conversion

    Returns:
        The model (converted if use_sync_batchnorm is True)
    """
    if use_sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model
