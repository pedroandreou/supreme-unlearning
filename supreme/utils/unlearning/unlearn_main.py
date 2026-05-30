# =============================================================================
# GPU BINDING FIX FOR SLURM
# Must be at the very top, before any torch/CUDA imports
# =============================================================================
import os as _os

def _fix_slurm_gpu_binding():
    """Fix GPU binding when SLURM doesn't set CUDA_VISIBLE_DEVICES per task.

    Some SLURM clusters allocate GPUs but don't set CUDA_VISIBLE_DEVICES per-task.
    This function uses SLURM_LOCALID to bind each task to its assigned GPU.
    Must be called BEFORE any torch/CUDA imports.
    """
    slurm_localid = _os.environ.get("SLURM_LOCALID")
    slurm_job_id = _os.environ.get("SLURM_JOB_ID")

    if slurm_job_id and slurm_localid is not None:
        cuda_visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "," in cuda_visible:  # Multiple GPUs visible - need to bind to one
            gpu_list = cuda_visible.split(",")
            local_id = int(slurm_localid)
            if local_id < len(gpu_list):
                assigned_gpu = gpu_list[local_id]
                _os.environ["CUDA_VISIBLE_DEVICES"] = assigned_gpu

_fix_slurm_gpu_binding()
# =============================================================================

from supreme.utils.generic_utils import set_seeds, initialize_network, dynamic_method_call
from supreme.utils.debug_utils import handle_distributed_error, create_debugger_session
from supreme.utils.unlearning.unlearning_utils import prepare_classwise_dataloaders
from supreme.utils.parsers.common_args import get_common_parser
import argparse
import os
from copy import deepcopy
from supreme.utils.model_logging import (
    save_model_and_logs,
    load_model_and_logs,
    check_model_files_exist,
    save_evaluation_results,
    get_model_save_path,
    save_logs_only,
)
from supreme.utils.memory_utils import cleanup, cleanup_unlearning_checkpoint
from lightning.fabric import Fabric
from supreme.utils.process_tracker import ProcessTracker
from supreme.utils.fabric.callbacks import (
    # checkpoint_callback,
    TrainingCallback,
    TestCallback,
    ParameterModificationCallback,
    MetricsEvaluationCallback,
)
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import warnings
import signal
import sys
from supreme.utils.unlearning.evaluation_utils import track_resources
from supreme.utils.fabric.fabric_setup import initialize_fabric, convert_to_sync_batchnorm, setup_model_for_inference
from supreme.utils.wandb_utils.runtime.wandb_setup import initialize_wandb, check_wandb_run_exists
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.fabric.accelerators.mps import MPSAccelerator
from supreme.eval_metrics.resource_consumption import (
    get_visible_gpu_ids,
)
import supreme.utils.project_config as project_config
from supreme.utils.project_config import (
    evaluation_metrics,
    metrics_requiring_retrain,
)
from typing import Optional
from supreme.eval_metrics.metrics_main import get_metric_scores
from supreme.utils.wandb_utils.runtime.wandb_setup import sync_wandb
from supreme.utils.generic_utils import strip_per_process_data

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"(TypedStorage is deprecated|Grad strides do not match bucket view strides)",
)


def requires_retrain(eval_metrics):
    """
    Validate the evaluation metrics and check if any require retrain.

    Args:
        eval_metrics (list): List of evaluation metric names

    Returns:
        bool: True if any metric requires retrain, False otherwise

    Raises:
        ValueError: If any metric is invalid
    """
    invalid_metrics = [m for m in eval_metrics if m not in evaluation_metrics]

    if invalid_metrics:
        raise ValueError(
            f"Invalid evaluation metrics: {invalid_metrics}. Valid metrics are: {', '.join(evaluation_metrics)}"
        )

    return any(m in metrics_requiring_retrain for m in eval_metrics)


def get_retrain_requiring_metrics(eval_metrics):
    """
    Get the list of evaluation metrics that require retrain.

    Args:
        eval_metrics (list): List of evaluation metric names

    Returns:
        list: List of metrics that require retrain
    """
    return [m for m in eval_metrics if m in metrics_requiring_retrain]


def validate_method_compatibility(
    method_name: str, type_of_unlearning_strategy: str
) -> None:
    """
    Validate if the unlearning method is compatible with the unlearning strategy.

    Args:
        method_name (str): Name of the unlearning method
        type_of_unlearning_strategy (str): Type of unlearning strategy (fullclass, subclass, or random_)

    Raises:
        ValueError: If the method is not compatible with the strategy
    """
    # Check UNSIR compatibility
    if type_of_unlearning_strategy == "random_" and method_name == "unsir":
        raise ValueError(
            "UNSIR method is not compatible with random_ unlearning strategy"
        )

    # Add any other method-strategy compatibility checks here


def setup_unlearning(
    fabric: Fabric,
    device: str,
    num_gpus: int,
    gpu_ids: str,
    seed: int,
    model_name: str,
    weight_path: str,
    dataset_name: str,
    batch_size: int,
    lr: float,
    type_of_unlearning_strategy: str,
    method_name: str,
    forget_class_name: str,
    forget_perc: float,
    forget_class_id: int,
    wandb_logging_flag: bool,
    export_class_distribution_info_flag: bool,
    precision: str,
    classes: Optional[int] = None,
    superclasses: Optional[int] = None,
    subclasses: Optional[int] = None,
    eval_metrics: Optional[list] = None,
    force_re_evaluation: bool = False,
    track_evaluation_resources: bool = False,
    force_reunlearning: bool = False,
    skip_evaluation_if_logged: bool = False,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    use_sync_batchnorm: bool = False,
    distributed_strategy_name: str = "ddp",
):
    # Validate method compatibility first
    validate_method_compatibility(method_name, type_of_unlearning_strategy)

    # Set seeds
    set_seeds(fabric, seed)

    # Get base network arguments
    base_net_kwargs = {
        "fabric": fabric,
        "model_name": model_name,
        "weight_path": weight_path,
        "device": device,
    }

    num_labels = None
    if (
        type_of_unlearning_strategy == "fullclass"
        or type_of_unlearning_strategy == "random_"
    ):
        num_labels = classes

        base_net_kwargs.update(
            {
                "num_labels": num_labels,
            }
        )
    elif type_of_unlearning_strategy == "subclass":
        num_labels = superclasses

        base_net_kwargs.update(
            {
                "num_labels": num_labels,
            }
        )
    fabric.print(
        f"\nInitializing the main network (original model) with weights from {weight_path}"
    )
    # Initialize the main network with weights
    original_model = initialize_network(**base_net_kwargs)
    fabric.print("The original model has been initialized successfully")

    # Initialize unlearning teacher only if bad_teacher or ZRF metric are requested
    needs_teacher = method_name == "bad_teacher"
    needs_zrf = eval_metrics and "zrf" in [metric.lower() for metric in eval_metrics]
    unlearning_teacher = None

    if needs_teacher or needs_zrf:
        message = "\nInitializing the unlearning teacher network without weights for "
        if needs_teacher and needs_zrf:
            message += "bad teacher method and ZRF metric"
        elif needs_teacher:
            message += "bad teacher method"
        else:  # only needs_zrf is True
            message += "ZRF metric"

        fabric.print(message)

        method_name_capitalized = "Unlearning_teacher"
        # =============================================== CHECK IF UNLEARNING TEACHER HAS ALREADY BEEN CREATED =============================================== #
        # Initialize the unlearning_teacher network without weights first
        unlearning_teacher_kwargs = base_net_kwargs.copy()
        unlearning_teacher_kwargs.pop(
            "weight_path", None
        )  # Remove weight_path for random initialization
        unlearning_teacher = initialize_network(**unlearning_teacher_kwargs)

        # Now check if files exist and load weights if they do
        files_exist = None  # Default value for all ranks
        if fabric.global_rank == 0:
            (
                unlearning_teacher,
                files_exist,
                _,
                _,
                _,
            ) = check_model_files_exist(
                fabric=fabric,
                model=unlearning_teacher,
                method_name=method_name_capitalized,
                device=device,
            )
        fabric.barrier()

        unlearning_teacher = fabric.broadcast(unlearning_teacher, src=0)
        files_exist = fabric.broadcast(files_exist, src=0)
        # ========================================================================================================================================================= #

        # IF UNLEARNING TEACHER HAS NOT ALREADY BEEN CREATED
        if not files_exist:
            unlearning_teacher = initialize_network(**unlearning_teacher_kwargs)

            # Extract state_dict before rank guard (consistent with FSDP-safe pattern)
            teacher_state_dict = unlearning_teacher.state_dict()
            if fabric.global_rank == 0:
                save_model_and_logs(
                    fabric=fabric,
                    method_name=method_name_capitalized,
                    state_dict=teacher_state_dict,
                    core_time_dict=None,
                    memory_usage_dict=None,
                    power_consumption_dict=None,
                )
            fabric.barrier()
            fabric.print("The unlearning teacher has been initialized successfully")
        else:
            fabric.print(
                "The unlearning teacher has already been initialized, so we will not initialize it again, but rather loaded from the checkpoint"
            )
        # ========================================================================================================================================================= #

    # It seems these variables are not broadcasted to all ranks
    export_class_distribution_info_flag = fabric.broadcast(
        export_class_distribution_info_flag, src=0
    )
    seed = fabric.broadcast(seed, src=0)
    forget_class_name = fabric.broadcast(forget_class_name, src=0)

    # Set up the dataloaders and prepare the datasets
    base_dataloader_kwargs = {
        "export_class_distribution_info_flag": export_class_distribution_info_flag,
        "seed": seed,
        "precision": precision,
        "num_gpus": num_gpus,
    }
    if type_of_unlearning_strategy == "random_":
        base_dataloader_kwargs.update(
            {
                "forget_class_name": forget_class_name,
                "forget_perc": forget_perc,
                "num_labels": forget_perc,
            }
        )
    elif type_of_unlearning_strategy == "fullclass":
        base_dataloader_kwargs.update(
            {
                "forget_class_id": forget_class_id,
                "forget_class_name": forget_class_name,
                "num_labels": classes,
            }
        )
    else:  # subclass
        base_dataloader_kwargs.update(
            {
                "forget_class_id": forget_class_id,
                "forget_class_name": forget_class_name,
                "num_labels": subclasses,
            }
        )

    loaders = prepare_classwise_dataloaders(
        fabric=fabric,
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        **base_dataloader_kwargs,
    )
    retain_train_dataloader = loaders["retain_train_dataloader"]
    retain_test_dataloader = loaders["retain_test_dataloader"]
    forget_train_dataloader = loaders["forget_train_dataloader"]
    forget_test_dataloader = loaders["forget_test_dataloader"]
    trainset = loaders["trainset"]
    testset = loaders["testset"]
    train_dataloader = loaders["train_dataloader"]
    test_dataloader = loaders["test_dataloader"]
    full_train_dataloader = loaders["full_train_dataloader"]
    retain_train_augmented_dataloader = loaders["retain_train_augmented_dataloader"]
    fabric.print("Prepared classwise dataloaders")

    forget_superclass_id = None
    if type_of_unlearning_strategy == "subclass":
        # Now iterate over the coarse_map in the test set
        if hasattr(testset, "coarse_map"):
            for idx, li in testset.coarse_map.items():  # type: ignore[attr-defined]
                if forget_class_id in li:
                    forget_superclass_id = idx
                    break
        else:
            raise ValueError("Testset does not have a coarse_map attribute")

        assert (
            forget_superclass_id is not None
        ), f"Subclass strategy failed: Could not find a superclass for class ID {forget_class_id}."

        fabric.print(
            "THE FORGET SUPERCLASS IS: ",
            forget_superclass_id,
            " where THE MATCHING FORGET CLASS ID IS: ",
            forget_class_id,
        )

    # model = None
    # retrained_model = None
    # if fabric.global_rank == 0:
    # Create deep copies of the original model
    model = deepcopy(
        original_model
    )  # Independent copy of the model so it can go under the unlearning procedure
    retrained_model = deepcopy(original_model)

    # Load the retrained model
    retrain_time_elapsed_dict = None
    retrain_memory_usage_dict = None
    retrain_power_consumption_dict = None
    fabric.print(f"The current method that is running is: '{method_name}'")

    # Only try to load retrained model if we need retrain and this isn't the retrain method
    if (
        # fabric.global_rank == 0
        # and
        method_name != "retrain" and requires_retrain(eval_metrics)
    ):
        fabric.print(
            "This method is not 'retrain' and evaluation metrics require retrain, so let's load the 'retrain' method."
        )
        # Load the retrained model and its time elapsed
        (
            retrained_model,
            retrain_time_elapsed_dict,
            retrain_memory_usage_dict,
            retrain_power_consumption_dict,
        ) = load_model_and_logs(
            fabric=fabric,
            net=retrained_model,
            method_name="Retrain",
            device=device,
        )
    # fabric.barrier()
    # retrained_model = fabric.broadcast(retrained_model, src=0)
    retrain_time_elapsed_dict = fabric.broadcast(retrain_time_elapsed_dict, src=0)
    # retrain_memory_usage_dict = fabric.broadcast(retrain_memory_usage_dict, src=0)
    # retrain_power_consumption_dict = fabric.broadcast(
    #     retrain_power_consumption_dict, src=0
    # )

    # We cannot do
    # model, original_model, unlearning_teacher, retrained_model = setup(model, original_model, unlearning_teacher, retrained_model)
    # but rather they need to be setup separately
    # otherwise, only the first model would be of
    # <class 'lightning.fabric.wrappers._FabricModule'>
    # and the rest would be of
    # <class 'lightning.fabric.wrappers.FabricResNet'>
    # or <class 'lightning.fabric.wrappers.FabricViT'>

    # print(f"Rank {fabric.global_rank}: model device: {next(model.parameters()).device}")

    # Convert BatchNorm to SyncBatchNorm for multi-GPU training (must be done before fabric.setup)
    model = convert_to_sync_batchnorm(model, use_sync_batchnorm)
    original_model = convert_to_sync_batchnorm(original_model, use_sync_batchnorm)
    if unlearning_teacher is not None:
        unlearning_teacher = convert_to_sync_batchnorm(unlearning_teacher, use_sync_batchnorm)
    if retrained_model is not None:
        retrained_model = convert_to_sync_batchnorm(retrained_model, use_sync_batchnorm)

    # The main model will be modified by unlearning methods (e.g. reset_parameters() in retrain,
    # parameter freezing in SSD, etc.) and then re-wrapped via fabric.setup(model, optimizer).
    # - DDP: fabric.setup(model) wraps it; methods unwrap via .module then re-wrap with optimizer
    # - FSDP: move to device only. FSDP flattens/shards parameters, so reset_parameters()
    #   corrupts weight shapes. The unlearning method will FSDP-wrap via fabric.setup(model, optimizer).
    # - DeepSpeed: move to device only. DeepSpeed requires a single engine with an optimizer.
    #   The unlearning method will create the engine via fabric.setup(model, optimizer).
    if distributed_strategy_name in ("ddp",):
        model = fabric.setup(model)
    else:
        model = model.to(fabric.device)
    fabric.print("Setup model")
    # print(f"Rank {fabric.global_rank}: model device: {next(model.parameters()).device}")

    # Inference-only models: use setup_model_for_inference which handles
    # DeepSpeed by moving to device without creating a DeepSpeed engine.
    original_model = setup_model_for_inference(fabric, original_model, distributed_strategy_name)
    fabric.print("Setup original model")

    # Only set up unlearning_teacher if it's not None
    if unlearning_teacher is not None:
        unlearning_teacher = setup_model_for_inference(fabric, unlearning_teacher, distributed_strategy_name)
        fabric.print("Setup unlearning teacher")
    else:
        fabric.print("Skipping unlearning_teacher setup as it is None")

    if retrained_model is not None:
        retrained_model = setup_model_for_inference(fabric, retrained_model, distributed_strategy_name)
        fabric.print("Setup retrained model")
    else:
        fabric.print("Skipping retrained model setup as it is None")

    # # Redundant broadcast but let's see if it helps
    # fabric.broadcast(model, src=0)
    # fabric.broadcast(original_model, src=0)
    # fabric.broadcast(unlearning_teacher, src=0)
    # fabric.broadcast(retrained_model, src=0)

    # Set up dataloaders with Fabric for device placement
    # Using dict-based approach so adding/removing dataloaders doesn't break positional indexing
    dataloaders_to_setup = {
        "retain_train_dataloader": retain_train_dataloader,
        "retain_test_dataloader": retain_test_dataloader,
        "forget_train_dataloader": forget_train_dataloader,
        "forget_test_dataloader": forget_test_dataloader,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "full_train_dataloader": full_train_dataloader,
    }
    if retain_train_augmented_dataloader is not None:
        dataloaders_to_setup["retain_train_augmented_dataloader"] = retain_train_augmented_dataloader

    setup_results = fabric.setup_dataloaders(*dataloaders_to_setup.values())
    setup_loaders = dict(zip(dataloaders_to_setup.keys(), setup_results))
    retain_train_dataloader = setup_loaders["retain_train_dataloader"]
    retain_test_dataloader = setup_loaders["retain_test_dataloader"]
    forget_train_dataloader = setup_loaders["forget_train_dataloader"]
    forget_test_dataloader = setup_loaders["forget_test_dataloader"]
    train_dataloader = setup_loaders["train_dataloader"]
    test_dataloader = setup_loaders["test_dataloader"]
    full_train_dataloader = setup_loaders["full_train_dataloader"]
    if retain_train_augmented_dataloader is not None:
        retain_train_augmented_dataloader = setup_loaders["retain_train_augmented_dataloader"]
    fabric.print("Setup dataloaders")


    # ============================================================================
    # STEP 1: BASE ARGUMENTS - Required for ALL unlearning methods
    # ============================================================================
    # These core arguments are automatically passed to every method in the framework
    kwargs = {
        "fabric": fabric,
        "wandb_logging_flag": wandb_logging_flag,
        "type_of_unlearning_strategy": type_of_unlearning_strategy,
        "model": model,  # Independent copy of the original model so it can go under the unlearning procedure
        "model_name": model_name,
        "num_gpus": num_gpus,
        "distributed_strategy_name": distributed_strategy_name,
    }


    # ============================================================================
    # STEP 2: STRATEGY-SPECIFIC ARGUMENTS - Based on unlearning task type
    # ============================================================================
    # Arguments that depend on the chosen unlearning strategy (fullclass/subclass/random_)
    # Add arguments specific to the type of unlearning strategy
    if type_of_unlearning_strategy == "fullclass":
        kwargs.update(
            {
                "num_labels": num_labels,
                "forget_label_id": forget_class_id,
            }
        )
    elif type_of_unlearning_strategy == "subclass":
        kwargs.update(
            {
                "num_labels": num_labels,
                "forget_label_id": forget_superclass_id,
            }
        )
    elif type_of_unlearning_strategy == "random_":
        kwargs.update(
            {
                "num_labels": num_labels,
            }
        )


    # ============================================================================
    # STEP 3: METHOD-SPECIFIC ARGUMENTS - Customize for each unlearning method
    # ============================================================================
    # Add your method's custom arguments here following the existing pattern
    # Example: elif method_name == "your_method":
    #             kwargs.update({"your_param": value, ...})
    if method_name == "random_labeling":
        kwargs.update(
            {
                "retain_train_dataloader": retain_train_dataloader,
                "retain_test_dataloader": retain_test_dataloader,
                "forget_train_dataloader": forget_train_dataloader,
            }
        )
    elif method_name == "original":
        pass  # No additional arguments needed
    elif method_name == "bad_teacher":
        kwargs.update(
            {
                "unlearning_teacher": unlearning_teacher,
                "retain_train_dataloader": retain_train_dataloader,
                "forget_train_dataloader": forget_train_dataloader,
            }
        )
    elif method_name == "finetune":
        kwargs.update(
            {
                "retain_train_dataloader": retain_train_dataloader,
                "retain_test_dataloader": retain_test_dataloader,
            }
        )
    elif method_name == "retrain":
        kwargs.update(
            {
                "retain_train_dataloader": retain_train_augmented_dataloader or retain_train_dataloader,
                "retain_test_dataloader": retain_test_dataloader,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_labels": num_labels,
                "lr": lr,
            }
        )
    elif method_name in ("ssd", "lfssd"):
        # Change alpha here as described in the SSD paper
        # Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" at https://arxiv.org/pdf/2308.07707
        # Paper: "LOSS-FREE MACHINE UNLEARNING" at https://arxiv.org/pdf/2402.19308

        # We found a discrepancy between the SSD and LFSSD papers.
        # The LFSSD paper, the SSD method has an alpha of 10 on ViT model on cifar20 class unlearning
        # whereas in the SSD paper itself, it uses an alpha of 5 for the ViT model
        # regardess though, we went to the latest published version of the paper which is the LFSSD paper

        model_size_scaler = 1 # alpha is 10
        dampening_constant = 1

        if type_of_unlearning_strategy == "fullclass":
            if dataset_name == "Cifar20":
                if method_name == "ssd":
                    if model_name == "ResNet18":
                        model_size_scaler = 1 # alpha is 10
                    else: # ViT
                        # this in SSD paper is 0.5 so alpha is 5
                        # but in LFSSD paper, it is 1 so alpha is 10
                        # we went to the latest published version of the paper which is the LFSSD paper
                        model_size_scaler = 1 # alpha is 10
                elif method_name == "lfssd":
                    # regardless of model architecture
                    model_size_scaler = 0.5 # alpha is 5

            elif dataset_name == "Cifar100":
                if method_name == "ssd":
                    if model_name == "ResNet18":
                        model_size_scaler = 1 # alpha is 10
                    else: # ViT
                        model_size_scaler = 1 # alpha is 10
                elif method_name == "lfssd":
                    # regardless of model architecture
                    model_size_scaler = 1 # alpha is 10

            elif dataset_name == "PinsFaceRecognition":
                if method_name == "ssd":
                    model_size_scaler = 5 # alpha is 50
                    dampening_constant = 0.1
                elif method_name == "lfssd":
                    # regardless of model architecture
                    model_size_scaler = 1 # alpha is 10

        elif type_of_unlearning_strategy == "subclass":
            # Cifar20 dataset
            if method_name == "ssd":
                if model_name == "ResNet18":
                    model_size_scaler = 1 # alpha is 10
                elif model_name == "ViT":
                    model_size_scaler = 2.5 # alpha is 25
            elif method_name == "lfssd":
                model_size_scaler = 1 # alpha is 10

        elif type_of_unlearning_strategy == "random_":
            if method_name == "ssd":
                model_size_scaler = 1 # alpha is 10
            elif method_name == "lfssd":
                model_size_scaler = 0.35 # alpha is 3.5

        # Calculate final alpha value
        selection_weighting = 10 * model_size_scaler

        # Print the selected hyperparameters and reasoning
        fabric.print(f"\n{'='*80}")
        fabric.print(f"SSD/LFSSD Hyperparameter Selection:")
        fabric.print(f"  Method: {method_name.upper()}")
        fabric.print(f"  Dataset: {dataset_name}")
        fabric.print(f"  Model: {model_name}")
        fabric.print(f"  Unlearning Strategy: {type_of_unlearning_strategy}")
        fabric.print(f"  Selected Alpha (selection_weighting): {selection_weighting}")
        fabric.print(f"  Selected Lambda (dampening_constant): {dampening_constant}")
        fabric.print(f"  Rationale: Based on hyperparameters from the {'LFSSD' if method_name == 'lfssd' else 'SSD (following LFSSD paper updates)'} paper")
        fabric.print(f"             for {dataset_name} dataset with {model_name} architecture")
        fabric.print(f"             under {type_of_unlearning_strategy} unlearning strategy")
        fabric.print(f"{'='*80}\n")

        kwargs.update(
            {
                "forget_train_dataloader": forget_train_dataloader,
                "full_train_dataloader": full_train_dataloader,
                "dampening_constant": dampening_constant,
                "selection_weighting": selection_weighting,
            }
        )
    elif method_name == "unsir":
        kwargs.update(
            {
                "trainset": trainset,
                "retain_train_dataloader": retain_train_dataloader,
                "retain_test_dataloader": retain_test_dataloader,
            }
        )
    elif method_name == "neg_grad":
        kwargs.update(
            {
                "forget_train_dataloader": forget_train_dataloader,
                "forget_test_dataloader": forget_test_dataloader,
            }
        )
    elif method_name == "assd":
        # Adaptive Selective Synaptic Dampening
        # Uses adaptive parameter selection based on importance distribution
        dampening_constant = 1.0
        selection_weighting = 10.0  # Initial value, will be adapted

        fabric.print(f"\n{'='*80}")
        fabric.print(f"ASSD (Adaptive SSD) Hyperparameter Selection:")
        fabric.print(f"  Dataset: {dataset_name}")
        fabric.print(f"  Model: {model_name}")
        fabric.print(f"  Unlearning Strategy: {type_of_unlearning_strategy}")
        fabric.print(f"  Initial Selection Weighting (Alpha): {selection_weighting} (will be adapted)")
        fabric.print(f"  Dampening Constant (Lambda): {dampening_constant}")
        fabric.print(f"  Rationale: ASSD automatically adapts alpha based on importance distribution")
        fabric.print(f"{'='*80}\n")

        kwargs.update(
            {
                "forget_train_dataloader": forget_train_dataloader,
                "full_train_dataloader": full_train_dataloader,
                "dampening_constant": dampening_constant,
                "selection_weighting": selection_weighting,
            }
        )

    elif method_name == "jit":
        # JIT (Just-In-Time) unlearning
        fabric.print(f"\n{'='*80}")
        fabric.print(f"JIT Unlearning Hyperparameter Selection:")
        fabric.print(f"  Dataset: {dataset_name}")
        fabric.print(f"  Model: {model_name}")
        fabric.print(f"  Unlearning Strategy: {type_of_unlearning_strategy}")
        fabric.print(f"  Epochs: 1 (default)")
        fabric.print(f"  Learning Rate: 0.001 (default)")
        fabric.print(f"  JIT Weighting: 0.1 (default)")
        fabric.print(f"  Rationale: JIT method enforces local smoothness constraints")
        fabric.print(f"{'='*80}\n")

        kwargs.update(
            {
                "forget_train_dataloader": forget_train_dataloader,
                # Optional: can override defaults with method-specific args
                # "n_epochs": 1,
                # "n_samples": 10,
                # "learning_rate": 0.001,
                # "jit_weighting": 0.1,
            }
        )

    elif method_name == "scrub":
        # SCRUB - Smoothed Gradient Descent-Ascent
        fabric.print(f"\n{'='*80}")
        fabric.print(f"SCRUB Hyperparameter Selection:")
        fabric.print(f"  Dataset: {dataset_name}")
        fabric.print(f"  Model: {model_name}")
        fabric.print(f"  Unlearning Strategy: {type_of_unlearning_strategy}")
        fabric.print(f"  SGDA Epochs: 10 (default)")
        fabric.print(f"  SGDA Learning Rate: 0.0005 (default)")
        fabric.print(f"  Gamma: 1.0 (default)")
        fabric.print(f"  Alpha: 0.5 (default)")
        fabric.print(f"  Rationale: SCRUB uses gradient ascent on forget set, descent on retain set")
        fabric.print(f"{'='*80}\n")

        kwargs.update(
            {
                "retain_train_dataloader": retain_train_dataloader,
                "forget_train_dataloader": forget_train_dataloader,
                # Optional: can override defaults with method-specific args
                # "sgda_epochs": 10,
                # "sgda_learning_rate": 0.0005,
                # "gamma": 1.0,
                # "alpha": 0.5,
            }
        )

    # [ADD YOUR NEW METHOD HERE]
    # elif method_name == "your_method":
    #     kwargs.update(
    #         {
    #             "your_custom_dataloader": your_dataloader,
    #             "temperature": 0.5,
    #             "alpha": 1.0,
    #         }
    #     )

    # Define the module path based on the method type
    if method_name in project_config.baselines:
        module_path = f"supreme.methods.baselines.{method_name}"
    elif method_name in project_config.unlearning_methods:
        module_path = f"supreme.methods.unlearning_methods.{method_name}"
    else:
        raise ValueError(f"Method {method_name} is not recognized.")

    method_name_capitalized = method_name.capitalize()

    # =============================================== CHECK IF MODEL HAS ALREADY BEEN UNLEARNED WITH THIS METHOD =============================================== #
    files_exist = None
    core_time_dict = None
    memory_usage_dict = None
    power_consumption_dict = None

    # Only check for existing files if force_reunlearning is False
    if not force_reunlearning:
        (
            model,
            files_exist,
            core_time_dict,
            memory_usage_dict,
            power_consumption_dict,
        ) = check_model_files_exist(
            fabric=fabric,
            model=model,
            method_name=method_name_capitalized,
            device=device,
        )

    else:
        # Force reunlearning: skip checkpoint check and set files_exist to False
        files_exist = False
        fabric.print(
            f"Force reunlearning enabled: Skipping checkpoint check for method '{method_name}'. "
            "Will regenerate unlearned model even if artifacts exist."
        )
    # ========================================================================================================================================================= #

    # IF MODEL HAS NOT ALREADY BEEN UNLEARNED WITH THIS METHOD
    if not files_exist:
        # Clear, prominent start banner for UNLEARNING phase
        fabric.print(
            "\n################################################################################\n"
            f"### STARTING UNLEARNING PHASE ###\n"
            f"### Method: {method_name.upper()} | Model: {model_name} | Dataset: {dataset_name} ###\n"
            f"### Strategy: {type_of_unlearning_strategy} | Forget: {forget_class_name} | Seed: {seed} ###\n"
            f"### World size: {fabric.world_size} | Local devices: {num_gpus} | Rank: {fabric.global_rank} ###\n"
            "################################################################################"
        )
        try:
            (
                method_result,
                core_time_dict,
                memory_usage_dict,
                power_consumption_dict,
            ) = track_resources(
                dynamic_method_call,
                module_name=module_path,
                file_name=method_name,
                **kwargs,
            )

        except Exception as e_unlearn:
            fabric.print(
                f"Error on rank {fabric.global_rank} during unlearning method execution: {str(e_unlearn)}"
            )
            raise

        if isinstance(method_result, torch.nn.Module):
            model = method_result

        # Save the unlearning model checkpoint. fabric.save() handles DDP/FSDP
        # unwrapping correctly when the model is actually wrapped. Two carve-outs:
        # - DeepSpeed: fabric.save() produces a sharded directory incompatible
        #   with torch.load(), so we save the raw state_dict via torch.save.
        # - FSDP parameter-surgery methods (SSD family, Importance-Freezing
        #   family) skip fabric.setup() and return a raw replicated nn.Module
        #   (see README: Parameter-surgery methods fall back to replicated mode
        #   under FSDP and DeepSpeed). fabric.save() would reject this because
        #   FSDP's save_checkpoint requires an FSDP-wrapped model, so we also
        #   use torch.save for this case.
        model_path = get_model_save_path(method_name_capitalized)
        if distributed_strategy_name.startswith("deepspeed"):
            if distributed_strategy_name == "deepspeed_stage3":
                # ZeRO-3 zero.Init makes each parameter shape [0] outside the
                # ZeRO partition table; gather them on rank 0 first so
                # state_dict() returns full-size tensors. Other ranks must
                # enter the same context (it's a collective gather).
                import deepspeed
                with deepspeed.zero.GatheredParameters(
                    list(model.parameters()), modifier_rank=0
                ):
                    if fabric.global_rank == 0:
                        raw_model = model.module if hasattr(model, "module") else model
                        torch.save(raw_model.state_dict(), model_path)
                        fabric.print(f'{method_name_capitalized} model saved to {model_path}')
            else:
                if fabric.global_rank == 0:
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), model_path)
                    fabric.print(f'{method_name_capitalized} model saved to {model_path}')
            fabric.barrier()
        elif distributed_strategy_name == "fsdp" and not any(
            isinstance(m, FSDP) for m in model.modules()
        ):
            if fabric.global_rank == 0:
                torch.save(model.state_dict(), model_path)
                fabric.print(f'{method_name_capitalized} model saved to {model_path}')
            fabric.barrier()
        else:
            # DDP, or FSDP with a properly-wrapped model.
            # All ranks must participate (FSDP needs collective gathering).
            fabric.save(model_path, {"model": model})
            fabric.print(f'{method_name_capitalized} model saved to {model_path}')

        # Save logs (time, memory, power) on rank 0 only
        if fabric.global_rank == 0:
            save_logs_only(
                fabric=fabric,
                method_name=method_name_capitalized,
                core_time_dict=core_time_dict,
                memory_usage_dict=memory_usage_dict,
                power_consumption_dict=power_consumption_dict,
            )
        fabric.barrier()

    else:
        if power_consumption_dict and "gpu_ids" in power_consumption_dict:
            # Normalize both GPU ID representations for comparison
            def normalize_gpu_ids(gpu_ids_value):
                """Convert GPU IDs to a consistent string format for comparison"""
                if isinstance(gpu_ids_value, (list, tuple)):
                    return ",".join(map(str, sorted(gpu_ids_value)))
                elif isinstance(gpu_ids_value, str):
                    # Remove brackets and spaces, then sort
                    clean_str = (
                        gpu_ids_value.replace("[", "").replace("]", "").replace(" ", "")
                    )
                    if clean_str:
                        gpu_list = [
                            int(x.strip()) for x in clean_str.split(",") if x.strip()
                        ]
                        return ",".join(map(str, sorted(gpu_list)))
                    return ""
                else:
                    return str(gpu_ids_value)

            original_gpus = normalize_gpu_ids(power_consumption_dict["gpu_ids"])
            current_gpus = normalize_gpu_ids(gpu_ids)

            if original_gpus != current_gpus:
                fabric.print(
                    f"Note: '{method_name}' originally used GPU ID(s) [{power_consumption_dict['gpu_ids']}] for unlearning but is now running on GPU ID(s) [{','.join(map(str, gpu_ids))}]. This difference doesn't affect the recorded memory and power consumption usage since they were measured during unlearning, not inference."
                )
            else:
                fabric.print(
                    f"Note: '{method_name}' originally used GPU ID(s) [{power_consumption_dict['gpu_ids']}] for unlearning and is now running on the same GPU ID(s) [{','.join(map(str, gpu_ids))}]. Memory and power consumption usage were measured during unlearning, not inference."
                )
        else:
            fabric.print(
                "Warning: Could not compare original and current GPUs because power consumption data is missing or incomplete."
            )

        kwargs["unlearned_model"] = kwargs.pop(
            "model"
        )  # Move model reference to unlearned_model

    kwargs.update(
        {
            "core_time_dict": core_time_dict,
        }
    )

    # ========================================================================================================================================================= #

    print(
        f"Rank {fabric.global_rank}: testset size = {len(testset)}, batches = {len(test_dataloader)}"
    )

    """
    Conditional Evaluation and Two-Phase Execution
    --------------------------------------------
    This script integrates both unlearning and evaluation, but they are designed to be
    run in two separate phases, controlled by the `PERFORM_EVALUATION` environment
    variable and a controlling script like `run_local.sh`.

    Phase 1: Unlearning (Multi-GPU)
    - The script is first run with `PERFORM_EVALUATION` set to "false" (or unset).
    - It performs the computationally intensive unlearning step, which can leverage
      multiple GPUs for efficiency.
    - The unlearned model and logs are saved to disk.

    Phase 2: Evaluation (Single-GPU)
    - The script is then run a second time with `PERFORM_EVALUATION` set to "true".
    - It reloads the models and data but executes the evaluation on a single GPU.

    This two-phase approach, while seeming inefficient due to reloading, is
    necessary for two critical reasons:

    1.  **Performance on Small Datasets**: Evaluation metrics are typically computed
        on small test sets. In a multi-GPU setup, the overhead of distributed
        processing on these small datasets makes evaluation significantly slower
        than running it on a single GPU.

    2.  **Technical Execution Constraints**: Attempting to run evaluation on only a
        single process (e.g., global_rank == 0) within a live multi-GPU unlearning
        script can cause the entire distributed process to hang.

    By separating the execution, we ensure both unlearning and evaluation run in
    their most optimal environments.
    """

    eval_result = None
    perform_evaluation = os.getenv("PERFORM_EVALUATION", "false").lower() == "true"
    if perform_evaluation:
        # Check if we should skip evaluation based on existing WandB logs
        if skip_evaluation_if_logged:
            fabric.print(
                f"\nChecking WandB for existing evaluation results (project: {project_name}, run: {run_name}, metrics: {eval_metrics})..."
            )
            status, missing_metrics = check_wandb_run_exists(project_name, run_name, eval_metrics)
            if status == "all_exist":
                fabric.print(
                    "\n################################################################################\n"
                    "### SKIPPING EVALUATION PHASE - Results already exist in WandB ###\n"
                    f"### Project: {project_name} ###\n"
                    f"### Run: {run_name} ###\n"
                    f"### Metrics: {eval_metrics} ###\n"
                    "################################################################################\n"
                )
                # Return early without performing evaluation
                return None
            elif status == "partial":
                fabric.print(
                    f"Partial metrics found in WandB. Only evaluating missing metrics: {missing_metrics}\n"
                )
                eval_metrics = missing_metrics
            else:
                fabric.print("No existing evaluation results found in WandB. Proceeding with evaluation.\n")

        # Clear, prominent start banner for EVALUATION phase (new subprocess run)
        fabric.print(
            "\n################################################################################\n"
            f"### STARTING EVALUATION PHASE ###\n"
            f"### Method: {method_name.upper()} | Model: {model_name} | Dataset: {dataset_name} ###\n"
            f"### Strategy: {type_of_unlearning_strategy} | Seed: {seed} ###\n"
            f"### World size: {fabric.world_size} | Local devices: {num_gpus} | Rank: {fabric.global_rank} ###\n"
            "################################################################################"
        )
        # Perform evaluation
        eval_kwargs = {
            "fabric": fabric,
            "num_gpus": num_gpus,
            "model_name": model_name,
            "forget_class_id": forget_class_id,
            "type_of_unlearning_strategy": type_of_unlearning_strategy,
            "eval_metrics": eval_metrics,
            "lr": lr,
            "batch_size": batch_size,
            "original_model": original_model,
            "unlearned_model": model,
            "unlearning_teacher": unlearning_teacher,
            "retrained_model": retrained_model
            if method_name != "retrain" and requires_retrain(eval_metrics)
            else None,
            "retrain_time_elapsed_dict": retrain_time_elapsed_dict
            if method_name != "retrain"
            else core_time_dict,
            "model_unlearned_with_initial_gpu_ids": power_consumption_dict.get(
                "gpu_ids"
            )
            if power_consumption_dict
            else None,
            "retain_train_dataloader": retain_train_dataloader,
            "retain_test_dataloader": retain_test_dataloader,
            "forget_train_dataloader": forget_train_dataloader,
            "forget_test_dataloader": forget_test_dataloader,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "trainset": trainset,
            "core_time_dict": core_time_dict,
            "memory_usage_dict": memory_usage_dict,
            "power_consumption_dict": power_consumption_dict,
            "wandb_logging_flag": wandb_logging_flag,
            "track_evaluation_resources": track_evaluation_resources,
        }

        # eval_result = None
        # if fabric.global_rank == 0:
        eval_result = get_metric_scores(**eval_kwargs)
        # fabric.barrier()

        # eval_result = fabric.broadcast(eval_result, src=0)

        # Always log during evaluation phase
        # Prepare the resource metrics from the UNLEARNING process to be logged at the top level
        unlearning_resource_metrics = {}
        if memory_usage_dict and power_consumption_dict:
            # Base resource metrics (always included)
            unlearning_resource_metrics = {
                "TotalGPUMemoryGB": memory_usage_dict["total_gpu_memory"],
                "TotalCPUMemoryGB": memory_usage_dict["total_cpu_memory"],
                "MaxGPUMemoryGB": memory_usage_dict["max_gpu_memory"],
                "MaxCPUMemoryGB": memory_usage_dict["max_cpu_memory"],
                "GPUIDs": power_consumption_dict["gpu_ids"],
                "StartSMUtilTotal": power_consumption_dict["start_sm_util"]["total"],
                "StartSMUtilMax": power_consumption_dict["start_sm_util"]["max"],
                "EndSMUtilTotal": power_consumption_dict["end_sm_util"]["total"],
                "EndSMUtilMax": power_consumption_dict["end_sm_util"]["max"],
                "TotalAverageSMUtil": power_consumption_dict["total_avg_sm_util"],
                "TotalPeakSMUtil": power_consumption_dict["total_peak_sm_util"],
                "MaxAverageSMUtil": power_consumption_dict["max_avg_sm_util"],
                "MaxPeakSMUtil": power_consumption_dict["max_peak_sm_util"],
                "TotalSMSeconds": power_consumption_dict["total_sm_seconds"],
                "TotalSMHours": power_consumption_dict["total_sm_hours"],
                "LogicalCPUCount": power_consumption_dict["logical_cpu_count"],
                "TotalAverageCPUUtil": power_consumption_dict["total_avg_cpu_util"],
                "TotalPeakCPUUtil": power_consumption_dict["total_peak_cpu_util"],
                "MaxAverageCPUUtil": power_consumption_dict["max_avg_cpu_util"],
                "MaxPeakCPUUtil": power_consumption_dict["max_peak_cpu_util"],
                "TotalCPUSeconds": power_consumption_dict["total_cpu_seconds"],
                "TotalCPUHours": power_consumption_dict["total_cpu_hours"],
            }

            # Conditionally add per-process metrics
            if os.getenv("LOG_PER_PROCESS_DATA", "false").lower() == "true":
                unlearning_resource_metrics.update(
                    {
                        "PerProcessGPUMemoryGB": memory_usage_dict["per_process"][
                            "gpu_memory"
                        ],
                        "PerProcessCPUMemoryGB": memory_usage_dict["per_process"][
                            "cpu_memory"
                        ],
                        "StartSMUtilPerProcess": power_consumption_dict["start_sm_util"][
                            "per_process"
                        ],
                        "EndSMUtilPerProcess": power_consumption_dict["end_sm_util"][
                            "per_process"
                        ],
                        "PerProcessAverageSMUtil": power_consumption_dict["per_process"][
                            "avg_sm_util"
                        ],
                        "PerProcessPeakSMUtil": power_consumption_dict["per_process"][
                            "peak_sm_util"
                        ],
                        "PerProcessSMSeconds": power_consumption_dict["per_process"][
                            "sm_seconds"
                        ],
                        "PerProcessSMHours": power_consumption_dict["per_process"][
                            "sm_hours"
                        ],
                        "PerProcessAverageCPUUtil": power_consumption_dict[
                            "per_process"
                        ]["avg_cpu_util"],
                        "PerProcessPeakCPUUtil": power_consumption_dict["per_process"][
                            "peak_cpu_util"
                        ],
                        "PerProcessCPUSeconds": power_consumption_dict["per_process"][
                            "cpu_seconds"
                        ],
                        "PerProcessCPUHours": power_consumption_dict["per_process"][
                            "cpu_hours"
                        ],
                    }
                )
        else:
            fabric.print(
                "Warning: Memory or power consumption data not available, skipping resource metric logging."
            )

        # Conditionally strip per-process data from evaluation results
        log_per_process = os.getenv("LOG_PER_PROCESS_DATA", "false").lower() == "true"

        # Strip per-process data if needed
        final_eval_result = (
            eval_result if log_per_process else strip_per_process_data(eval_result)
        )

        # Add unlearning resource metrics to the final result
        # Create a new "unlearning_resources" section in the result
        if unlearning_resource_metrics:
            if final_eval_result is None:
                final_eval_result = {}
            for key, value in unlearning_resource_metrics.items():
                final_eval_result[key] = value

        # Now log or print the combined result
        if wandb_logging_flag:
            if fabric.global_rank == 0:
                # Single logging point with all metrics combined
                fabric.log_dict(final_eval_result)
                sync_wandb(fabric)
            fabric.barrier()
        else:
            try:
                fabric.print("====== METRICS SUMMARY ======")
                fabric.print(f"eval_result: {final_eval_result}")
                fabric.print("====== END OF METRICS ======")
            except Exception as e:
                fabric.print(f"Error during metrics printing: {e}")

        # Save evaluation results locally as JSON (append if file already exists)
        if fabric.global_rank == 0 and final_eval_result:
            try:
                save_evaluation_results(
                    fabric=fabric,
                    method_name=method_name_capitalized,
                    eval_result=final_eval_result,
                )
            except Exception as e:
                fabric.print(f"Warning: Failed to save evaluation results to JSON: {e}")
    else:
        fabric.print(
            "Evaluation step skipped (PERFORM_EVALUATION=false). This run performed only the unlearning stage."
        )
        eval_result = None

    # Return results as well as variables for cleanup
    return {
        "original_model": original_model,
        "unlearning_teacher": unlearning_teacher,
        "retrained_model": retrained_model,
        "retrain_time_elapsed_dict": retrain_time_elapsed_dict,
        "retain_train_dataloader": retain_train_dataloader,
        "retain_test_dataloader": retain_test_dataloader,
        "forget_train_dataloader": forget_train_dataloader,
        "forget_test_dataloader": forget_test_dataloader,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "full_train_dataloader": full_train_dataloader,
        "trainset": trainset,
        "testset": testset,
        "eval_result": eval_result,
    }


def segfault_handler(signum, frame):
    """Handle segmentation faults by printing debug info"""
    print(f"\n{'='*80}")
    print("SEGMENTATION FAULT DETECTED!")
    print(f"Signal: {signum}")
    print(f"Frame: {frame}")
    print("Current process info:")
    print(f"  PID: {os.getpid()}")
    print("  Available GPU memory:")
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                print(
                    f"    GPU {i}: {mem_info[0] / 1e9:.2f}GB free / {mem_info[1] / 1e9:.2f}GB total"
                )
        elif torch.backends.mps.is_available():
            allocated_gb = torch.mps.current_allocated_memory() / 1e9
            driver_gb = torch.mps.driver_allocated_memory() / 1e9
            print(f"    MPS: {allocated_gb:.2f}GB tensor / {driver_gb:.2f}GB driver allocated")
    except Exception as e:
        print(f"    Could not get GPU memory info: {e}")
    print(f"{'='*80}")
    sys.exit(1)


def main():
    # Set up segfault handler
    signal.signal(signal.SIGSEGV, segfault_handler)

    # Step 1: Minimal parser to extract type_of_unlearning_strategy and dataset name
    initial_parser = argparse.ArgumentParser()
    initial_parser.add_argument(
        "-type_of_unlearning_strategy",
        type=str,
        required=True,
        help="Type of unlearning strategy: fullclass, subclass, or random_",
    )
    initial_parser.add_argument(
        "-dataset",
        type=str,
        required=True,
        choices=project_config.dataset_names,
        help="Dataset to train on",
    )
    initial_args, remaining_args = initial_parser.parse_known_args()

    # Step 2: Get common parser and add remaining arguments dynamically
    parser = get_common_parser()

    type_of_unlearning_strategy = initial_args.type_of_unlearning_strategy
    dataset_name = initial_args.dataset

    # Add dynamic arguments based on the type_of_unlearning_strategy
    class_dict = None

    if type_of_unlearning_strategy != "random_":
        if type_of_unlearning_strategy == "fullclass":
            parser.add_argument(
                "-classes", type=int, required=True, help="number of classes"
            )
            parser.add_argument(
                "-batch_size", type=int, default=64, help="batch size for dataloader"
            )

            # Get dictionary name from centralized config
            try:
                dict_name = project_config.get_dict_name_for_dataset(dataset_name, type_of_unlearning_strategy)
                class_dict = getattr(project_config, dict_name)
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Could not load class dictionary for dataset '{dataset_name}': {e}")

            parser.add_argument(
                "-forget_class_name",
                type=str,
                required=True,
                help="class to forget",
                choices=list(class_dict),
            )

        elif type_of_unlearning_strategy == "subclass":
            parser.add_argument(
                "-superclasses", type=int, required=True, help="number of superclasses"
            )
            parser.add_argument(
                "-subclasses", type=int, required=True, help="number of subclasses"
            )
            parser.add_argument(
                "-batch_size", type=int, default=64, help="batch size for dataloader"
            )

            # Get dictionary name from centralized config (subclass strategy uses CIFAR100 classes)
            try:
                dict_name = project_config.get_dict_name_for_dataset(dataset_name, type_of_unlearning_strategy)
                class_dict = getattr(project_config, dict_name)
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Could not load class dictionary for dataset '{dataset_name}' with strategy '{type_of_unlearning_strategy}': {e}")

            parser.add_argument(
                "-forget_subclass_name",
                type=str,
                required=True,
                help="class to forget",
                choices=list(class_dict),
            )

    else:  # "random_"
        parser.add_argument(
            "-classes", type=int, required=True, help="number of classes"
        )
        parser.add_argument(
            "-forget_perc",
            type=float,
            required=True,
            help="Percentage of set to forget",
        )
        parser.add_argument(
            "-batch_size", type=int, default=128, help="batch size for dataloader"
        )

    # Common arguments that are always required
    parser.add_argument(
        "-weight_path",
        type=str,
        required=True,
        help="Path to model weights. If you need to train a new model use train_main.py",
    )
    parser.add_argument(
        "-method",
        type=str,
        required=True,
        choices=project_config.all_methods,  # Use the combined list from project_config.py
        help="select unlearning method from choice set",
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=1,
        help="number of epochs of unlearning method to use",
    )
    parser.add_argument("-seed", type=int, default=0, help="seed for runs")

    parser.add_argument(
        "-precision",
        type=str,
        required=True,
        choices=[
            "32-true",
            "16-mixed",
            "16-true",
            "bf16-mixed",
            "bf16-true",
            "transformer-engine",
            "transformer-engine-float16",
            "64-true",
            "nf4",
            "nf4-dq",
            "fp4",
            "fp4-dq",
            "int8",
            "int8-training",
        ],
        help="Precision for training (32-true or 16-mixed or 16-true or bf16-mixed or bf16-true or transformer-engine or transformer-engine-float16 or 64-true or nf4 or nf4-dq or fp4 or fp4-dq or int8 or int8-training)",
    )

    parser.add_argument(
        "-eval_metrics",
        type=str,
        required=True,
        help="Comma-separated list of evaluation metrics to run",
    )

    parser.add_argument(
        "-force_re_evaluation",
        action="store_true",
        default=False,
        help="Force re-evaluation and logging even when models already exist (default: False)",
    )

    parser.add_argument(
        "-track_evaluation_resources",
        action="store_true",
        default=False,
        help="Track resource consumption (time/memory/power) during evaluation metrics computation (default: False)",
    )

    parser.add_argument(
        "-force_reunlearning",
        action="store_true",
        default=False,
        help="Force re-unlearning even when checkpoints exist (default: False)",
    )

    parser.add_argument(
        "-skip_evaluation_if_logged",
        action="store_true",
        default=False,
        help="Skip evaluation if results already exist in WandB for this configuration (default: False)",
    )

    parser.add_argument(
        "-cleanup_checkpoints_after_eval",
        action="store_true",
        default=False,
        help="Delete model checkpoint files after evaluation completes to save disk space (default: False)",
    )

    # Final parsing with all arguments
    args = parser.parse_args(remaining_args)

    model_name = args.net
    weight_path = args.weight_path
    precision = args.precision
    # In SLURM DDP mode with GPU binding:
    # - Each task sees only 1 GPU (CUDA_VISIBLE_DEVICES set per task)
    # - fabric_devices = 1 (what Fabric can actually use per task)
    # - num_gpus = world_size = SLURM_NTASKS (for paths like "4gpus/")
    fabric_devices = 1 if MPSAccelerator.is_available() else num_cuda_devices()  # What each task actually sees (1 with GPU binding)
    slurm_ntasks = _os.environ.get("SLURM_NTASKS")
    if slurm_ntasks:
        num_gpus = int(slurm_ntasks)  # World size for paths/logging
    else:
        num_gpus = fabric_devices
    gpu_ids = get_visible_gpu_ids()

    batch_size = args.batch_size
    lr = args.lr
    method_name = args.method.lower()
    seed = args.seed
    wandb_logging_flag = args.wandb_logging_flag
    tensorboard_logging_flag = args.tensorboard_logging_flag
    csv_logging_flag = args.csv_logging_flag
    logging_root_dir = args.logging_root_dir
    logging_enabled = wandb_logging_flag or tensorboard_logging_flag or csv_logging_flag
    export_class_distribution_info_flag = args.export_class_distribution_info_flag
    use_process_tracker = args.use_process_tracker
    force_re_evaluation = args.force_re_evaluation
    track_evaluation_resources = args.track_evaluation_resources
    force_reunlearning = args.force_reunlearning
    skip_evaluation_if_logged = args.skip_evaluation_if_logged
    cleanup_checkpoints_after_eval = args.cleanup_checkpoints_after_eval
    eval_metrics = args.eval_metrics.split(",")

    classes = (
        args.classes
        if type_of_unlearning_strategy in ("fullclass", "random_")
        else None
    )
    superclasses = (
        args.superclasses if type_of_unlearning_strategy == "subclass" else None
    )
    subclasses = args.subclasses if type_of_unlearning_strategy == "subclass" else None

    # Initialize variables
    tracker = None
    fabric = None
    returned_variables = None
    success = True

    try:
        # Conditionally prepare callbacks
        callbacks = []
        if os.getenv("USE_FABRIC_CALLBACKS", "false").lower() == "true":
            callbacks = [
                TrainingCallback(logging_enabled),
                TestCallback(logging_enabled),
                ParameterModificationCallback(logging_enabled),
                MetricsEvaluationCallback(logging_enabled),
            ]

        # Initialize Fabric
        # Use fabric_devices (what each task sees) for Fabric, not num_gpus (world_size)
        fabric_config = {
            "model_name": model_name,
            "precision": precision,
            "num_gpus": fabric_devices,
            "callbacks": callbacks,
            "distributed_strategy": args.distributed_strategy,
            "deepspeed_stage": args.deepspeed_stage,
            "tensorboard_logging_flag": tensorboard_logging_flag,
            "csv_logging_flag": csv_logging_flag,
            "logging_root_dir": logging_root_dir,
            "logging_run_name": f"{model_name}_{dataset_name}_{method_name}_precision_{precision}",
        }

        fabric, device, fabric_strategy, use_sync_batchnorm, distributed_strategy_name = initialize_fabric(fabric_config)

        if use_process_tracker:
            tracker = ProcessTracker(
                fabric=fabric,
                script_type="unlearn",
                model_name=model_name,
                type_of_unlearning_strategy=type_of_unlearning_strategy,
                dataset_name=dataset_name,
                num_gpus=num_gpus,
                batch_size=batch_size,  # this does not need scaling because is for each process
            )

        # Define the forget class
        forget_class_id = None
        forget_class_name = None
        forget_perc = (
            args.forget_perc if type_of_unlearning_strategy == "random_" else None
        )

        if fabric.global_rank == 0:
            if type_of_unlearning_strategy != "random_":
                if type_of_unlearning_strategy == "fullclass":
                    forget_class_name = args.forget_class_name
                else:  # subclass
                    forget_class_name = args.forget_subclass_name

                assert (
                    class_dict is not None
                ), "class_dict must be defined for non-random unlearning strategies"
                assert (
                    forget_class_name in class_dict
                ), f"{forget_class_name} not found in the dictionary of {dataset_name} dataset"

                forget_class_id = class_dict[forget_class_name]

                fabric.print(
                    f"Loaded forget class ID: {forget_class_id} with forget class name: {forget_class_name}"
                )
            else:  # random_
                forget_class_name = f"samples_{str(forget_perc)}"
        fabric.barrier()

        forget_class_id = fabric.broadcast(forget_class_id, src=0)
        forget_class_name = fabric.broadcast(forget_class_name, src=0)
        forget_perc = fabric.broadcast(forget_perc, src=0)

        # Get experiment type from environment variable
        experiment_scenario = os.getenv("SCALABLE_EXPERIMENT_SCENARIO", "")

        # Initialize WandB
        # Filter out empty strings from the list of parts to avoid double underscores
        # Use WANDB_PROJECT_PREFIX environment variable (default: R14) for flexibility
        wandb_project_prefix = os.getenv("WANDB_PROJECT_PREFIX", "R14")
        project_name_parts = list(
            filter(
                None,
                [
                    f"{wandb_project_prefix}_UNLEARNING",
                    experiment_scenario,
                    model_name,
                    dataset_name,
                    type_of_unlearning_strategy,
                    (
                        forget_class_name
                        if type_of_unlearning_strategy != "random_"
                        else f"{args.forget_perc}perc"
                    ),
                    f"precision_{precision}",
                    f"dist_{distributed_strategy_name}" if num_gpus > 1 else "no_dist",
                    # f"seed_{seed}", # no need for this so all different runs are in the same group
                    # f"{num_gpus}gpus",
                ],
            )
        )
        project_name = "_".join(project_name_parts)

        # Construct primary run name + a fallback list. The naming convention
        # evolved in three steps:
        #   1. `{method}_seed{U}`                          (oldest, no TRAINING_SEED)
        #   2. `{method}_tseed{T}_useed{U}`                (J>1, K=1: -seed = s_u)
        #   3. `{method}_tseed{T}_useed{U}_eseed{E}`       (K>1 eval: -seed = s_e ≠ s_u)
        # MAIN.sh exports UNLEARNING_SEED=s_u for every cell. When the active
        # -seed CLI arg differs from UNLEARNING_SEED, we know this is a K>1
        # evaluation run and emit the triple-form name. Earlier names are kept
        # as alt_run_names so wandb_setup can resume runs logged under any of
        # the older conventions.
        _training_seed_env = os.environ.get('TRAINING_SEED')
        _unlearning_seed_env = os.environ.get('UNLEARNING_SEED')
        if _training_seed_env and _unlearning_seed_env and int(_unlearning_seed_env) != int(seed):
            # K>1 evaluation: seed holds s_e, UNLEARNING_SEED holds s_u
            wandb_run_name = (
                f"{method_name}_tseed{_training_seed_env}"
                f"_useed{_unlearning_seed_env}_eseed{seed}"
            )
            wandb_alt_run_names = []
        elif _training_seed_env:
            # Matched or J>1/K=1: seed holds s_u (= s_e by collapse)
            wandb_run_name = f"{method_name}_tseed{_training_seed_env}_useed{seed}"
            wandb_alt_run_names = [f"{method_name}_seed{seed}"]
        else:
            wandb_run_name = f"{method_name}_seed{seed}"
            wandb_alt_run_names = []

        wandb_config = {
            "wandb_logging_flag": wandb_logging_flag,
            "project_name": project_name,
            "run_name": wandb_run_name,
            "alt_run_names": wandb_alt_run_names,
            "group_name": f"{method_name}_group",
            "experiment_config": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "unlearning_strategy": type_of_unlearning_strategy,
                "forget_class": forget_class_name,
                "seed": seed,
                "num_gpus": num_gpus,
                "gpu_ids": gpu_ids,
                "precision": precision,
                "distributed_strategy": distributed_strategy_name,
            },
            # Override WandB auto-detected GPU metrics with actual values
            # This ensures WandB reports the GPUs actually used, not all visible GPUs
            # See: https://docs.wandb.ai/ref/python/experiments/settings/
            "actual_gpu_count": num_gpus,
            "actual_gpu_ids": gpu_ids if isinstance(gpu_ids, list) else list(range(num_gpus)),
        }
        # Resume existing WandB run if requested (to append new metrics)
        # Set WANDB_RESUME_EXISTING=true to resume an existing run by name
        # instead of creating a new one. Useful for re-evaluating specific metrics.
        wandb_resume_existing = os.getenv("WANDB_RESUME_EXISTING", "false").lower() == "true"
        if wandb_resume_existing:
            wandb_config["resume_if_exists"] = True

        # Initialize WandB only during evaluation phase to prevent empty runs
        # Control via environment variables (consistent with PERFORM_EVALUATION pattern):
        #   WANDB_LOG_EVALUATION=true   - Log evaluation metrics to W&B (default: uses wandb_logging_flag)
        #   WANDB_LOG_UNLEARNING=false  - Log unlearning metrics to W&B (reserved for future use)
        perform_evaluation = os.getenv("PERFORM_EVALUATION", "false").lower() == "true"
        wandb_log_evaluation = os.getenv("WANDB_LOG_EVALUATION", str(wandb_logging_flag).lower()).lower() == "true"

        should_init_wandb = perform_evaluation and wandb_log_evaluation and fabric.global_rank == 0

        if should_init_wandb:
            fabric = initialize_wandb(fabric, wandb_config)
        elif fabric.global_rank == 0 and wandb_logging_flag and not perform_evaluation:
            fabric.print("Skipping WandB initialization during unlearning phase (no metrics to log yet)")

        fabric.barrier()

        debugger_session = None
        if fabric.global_rank == 0 and os.getenv("DEBUGGER"):
            debugger_session = create_debugger_session()
        fabric.barrier()

        # Track worker processes
        if (
            fabric.global_rank != 0
        ) and use_process_tracker:  # If this is a worker process (not the main process)
            if tracker:
                tracker.add_child_pid(os.getpid())

        main_params = {
            "fabric": fabric,
            "device": device,
            "num_gpus": num_gpus,
            "gpu_ids": gpu_ids,
            "seed": seed,
            "model_name": model_name,
            "weight_path": weight_path,
            "dataset_name": dataset_name,
            "batch_size": batch_size,
            "lr": lr,
            "type_of_unlearning_strategy": type_of_unlearning_strategy,
            "method_name": method_name,
            "forget_class_name": forget_class_name,
            "forget_perc": forget_perc,
            "forget_class_id": forget_class_id,
            "wandb_logging_flag": wandb_logging_flag,
            "export_class_distribution_info_flag": export_class_distribution_info_flag,
            "classes": classes,
            "superclasses": superclasses,
            "subclasses": subclasses,
            "precision": precision,
            "eval_metrics": eval_metrics,
            "force_re_evaluation": force_re_evaluation,
            "track_evaluation_resources": track_evaluation_resources,
            "force_reunlearning": force_reunlearning,
            "skip_evaluation_if_logged": skip_evaluation_if_logged,
            "project_name": project_name,
            "run_name": wandb_config["run_name"],
            "use_sync_batchnorm": use_sync_batchnorm,
            "distributed_strategy_name": distributed_strategy_name,
        }

        if debugger_session:
            with debugger_session:
                returned_variables = setup_unlearning(**main_params)
        else:
            returned_variables = setup_unlearning(**main_params)

    except Exception as e:
        success = False
        handle_distributed_error(fabric, e)

    finally:
        # Determine if we're running evaluation or unlearning
        perform_evaluation = os.getenv("PERFORM_EVALUATION", "false").lower() == "true"

        if perform_evaluation:
            message = (
                "\n################################################################################\n"
                "### EVALUATION completed successfully. Starting cleanup... ###\n"
                "################################################################################\n\n\n"
                if success
                else (
                    "\n################################################################################\n"
                    "### EVALUATION completed unsuccessfully. Starting cleanup... ###\n"
                    "################################################################################\n\n\n"
                )
            )
        else:
            message = (
                "\n################################################################################\n"
                "### UNLEARNING completed successfully. Starting cleanup... ###\n"
                "################################################################################\n\n\n"
                if success
                else (
                    "\n################################################################################\n"
                    "### UNLEARNING completed unsuccessfully. Starting cleanup... ###\n"
                    "################################################################################\n\n\n"
                )
            )
        
        fabric.print(message)

        # Cleanup model checkpoint after evaluation to save disk space
        if perform_evaluation and success and cleanup_checkpoints_after_eval:
            cleanup_unlearning_checkpoint(fabric, method_name, cleanup_enabled=True)

        # Cleanup to prevent segfaults during distributed exit
        cleanup(fabric, returned_variables)

        # Process tracker cleanup
        if use_process_tracker and tracker:
            try:
                tracker.cleanup()
            except Exception as e:
                print(f"Process tracker cleanup warning: {e}")


if __name__ == "__main__":  # needed for multiprocessing
    """
    Code structure style is suggested at
    https://lightning.ai/docs/fabric/stable/fundamentals/code_structure.html
    """
    main()
