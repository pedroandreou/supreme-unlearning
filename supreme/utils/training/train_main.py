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

import torch

import torch.optim as optim
import torch.nn as nn
from supreme.utils.training.training_utils import WarmUpLR
from supreme.utils.unlearning.unlearning_utils import prepare_dataloaders
from supreme.utils.generic_utils import initialize_network, set_seeds
import os
import supreme.utils.project_config as project_config
from supreme.utils.parsers.common_args import get_common_parser
from supreme.utils.debug_utils import handle_distributed_error, create_debugger_session
from supreme.utils.memory_utils import cleanup
from supreme.utils.process_tracker import ProcessTracker
from supreme.utils.fabric.callbacks import (
    # checkpoint_callback,
    TrainingCallback,
    TestCallback,
)
from supreme.utils.fabric.fabric_setup import (
    initialize_fabric,
    convert_to_sync_batchnorm,
)
from supreme.utils.wandb_utils.runtime.wandb_setup import initialize_wandb, sync_wandb
from lightning.fabric.accelerators.cuda import num_cuda_devices
from lightning.fabric.accelerators.mps import MPSAccelerator
import warnings
from typing import Optional

# Filter multiple warnings in a single call
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"(TypedStorage is deprecated|Grad strides do not match bucket view strides|The default value of the antialias parameter of all the resizing transforms.*)",
)


def train(
    fabric,
    model,
    optimizer,
    train_dataloader,
    loss_function,
    warm,
    warmup_scheduler,
    epoch,
):
    # if fabric.global_rank == 0:
    #     fabric.call(
    #         "on_train_epoch_start",
    #         fabric=fabric,
    #         epoch=epoch,
    #     )
    # fabric.barrier()

    model.train()
    loss = 0.0
    for batch_index, (images, _, labels) in enumerate(train_dataloader):
        # if fabric.global_rank == 0:
        #     fabric.call("on_train_batch_start")
        # fabric.barrier()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        if epoch <= warm:
            warmup_scheduler.step()

        # if fabric.global_rank == 0:
        #     fabric.call(
        #         "on_train_batch_end",
        #         loss=loss,
        #         epoch=epoch,
        #         batch_idx=batch_index,
        #         lr=optimizer.param_groups[0]["lr"],
        #     )
        # fabric.barrier()

    # if fabric.global_rank == 0:
    #     # Call epoch end
    #     fabric.call(
    #         "on_train_epoch_end",
    #         epoch=epoch,
    #         train_loss=loss,
    #         last_lr=optimizer.param_groups[0]["lr"],
    #     )
    # fabric.barrier()


@torch.no_grad()
def eval_training(fabric, model, test_dataloader, loss_function, epoch=0):
    # if fabric.global_rank == 0:
    #     fabric.call("on_test_epoch_start", fabric=fabric)
    # fabric.barrier()

    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for batch_idx, (images, _, labels) in enumerate(test_dataloader):
        # if fabric.global_rank == 0:
        #     fabric.call("on_test_batch_start")
        # fabric.barrier()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        batch_correct = preds.eq(labels).sum()
        correct += batch_correct

        # if fabric.global_rank == 0:
        #     fabric.call(
        #         "on_test_batch_end",
        #         loss=loss,
        #         epoch=epoch,
        #         batch_idx=batch_idx,
        #         acc=100 * batch_correct.float() / labels.size(0),
        #     )
        # fabric.barrier()

    # ############################################################
    # Aggregate metrics across all processes
    test_loss = fabric.all_gather(test_loss).sum() / len(test_dataloader.dataset)
    correct = fabric.all_gather(correct).sum() / len(test_dataloader.dataset)
    # ############################################################

    # Only print GPU info and results on rank 0 to avoid duplicate outputs
    # fabric.print("GPU INFO.....")
    # fabric.print(torch.cuda.memory_summary(), end="")

    # if fabric.global_rank == 0:
    #     # Call test end
    #     fabric.call(
    #         "on_test_epoch_end",
    #         epoch=epoch,
    #         loss=test_loss,
    #         acc=correct,
    #     )
    # fabric.barrier()

    return correct


def setup_training(
    fabric,
    num_gpus: int,
    model_name: str,
    dataset_name: str,
    class_num: int,
    batch_size: int,
    lr: float,
    device: str,
    warm: int,
    MILESTONES: tuple,
    EPOCHS: int,
    precision: str,
    training_seed: Optional[int] = None,
    unlearning_seed: Optional[int] = None,
    unlearning_context: str = "N/A",
    include_gpus_in_path: bool = True,
    use_sync_batchnorm: bool = False,
    distributed_strategy_name: str = "ddp",
):
    # Clear, prominent start banner for TRAINING phase
    fabric.print(
        "\n################################################################################\n"
        f"### STARTING TRAINING PHASE ###\n"
        f"### Model: {model_name} | Dataset: {dataset_name} | Classes: {class_num} ###\n"
        f"### Precision: {precision} | Seed: {unlearning_seed} ###\n"
        f"### World size: {fabric.world_size} | Local devices: {num_gpus} | Rank: {fabric.global_rank} ###\n"
        f"### Distributed strategy: {distributed_strategy_name} ###\n"
        "################################################################################"
    )
    if training_seed:
        fabric.print(f"Using training seed {training_seed} for reproducible training")
        set_seeds(fabric, training_seed)
    else:
        fabric.print("No training seed provided, using random initialization")

    # Get network
    model = initialize_network(
        fabric=fabric, model_name=model_name, num_labels=class_num, device=device
    )

    # Set up the Dataloaders
    trainset, testset, train_dataloader, test_dataloader = prepare_dataloaders(
        fabric=fabric,
        num_gpus=num_gpus,
        precision=precision,
        model_name=model_name,
        seed=unlearning_seed,  # Pass the seed to dataloaders
        dataset_name=dataset_name,
        type_of_unlearning_strategy=None,
        batch_size=batch_size,
        unlearning=False,
        export_class_distribution_info_flag=False,
    )

    # Prepare training components
    loss_function = nn.CrossEntropyLoss(reduction="mean")
    if model_name == "ViT":
        vit_lr = getattr(project_config, "ViT_LR", 5e-5)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=vit_lr,
            weight_decay=0.01,
        )
        fabric.print(f"ViT optimizer: AdamW (lr={vit_lr}, weight_decay=0.01)")
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

    # Convert BatchNorm to SyncBatchNorm for multi-GPU training (must be done before fabric.setup)
    model = convert_to_sync_batchnorm(model, use_sync_batchnorm)

    # Create ALL LR schedulers BEFORE fabric.setup() - DeepSpeed wraps the optimizer
    # in FabricDeepSpeedZeroOptimizer which is not recognized by PyTorch schedulers.
    # WarmUpLR also extends _LRScheduler and has the same issue.
    if model_name == "ViT":
        train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - warm
        )
        fabric.print(f"ViT scheduler: CosineAnnealingLR (T_max={EPOCHS - warm})")
    else:
        train_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=MILESTONES, gamma=0.2
        )  # learning rate decay

    # Compute iter_per_epoch from raw dataloader (before fabric wraps it)
    iter_per_epoch = len(train_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warm)

    # Use Fabric to move both models and data to the device
    model, optimizer = fabric.setup(model, optimizer)

    # Already setup in prepare_dataloaders
    # Set them up there so we can export the class distribution info
    # as we would face errors with device mismatch
    train_dataloader, test_dataloader = fabric.setup_dataloaders(
        train_dataloader, test_dataloader
    )

    ##########################################################
    #### PREPARE FOR PRETRAINING AND PRODUCING CHECKPOINTS ###
    ##########################################################

    # Construct the checkpoint path with the lowercase dataset name
    # Always include seed in the path (use "None" if not provided)
    # fabric.print(f"[Debug] Received seeds for path construction: training_seed='{training_seed}' (type: {type(training_seed)}), unlearning_seed='{unlearning_seed}' (type: {type(unlearning_seed)})")
    training_seed_str = (
        f"train_seed_{'none' if training_seed is None else training_seed}"
    )
    unlearning_seed_str = (
        f"unlearning_seed_{'none' if unlearning_seed is None else unlearning_seed}"
    )
    gpu_str = f"{num_gpus}gpus" if include_gpus_in_path else ""
    dist_str = f"dist_{distributed_strategy_name}" if num_gpus > 1 else "no_dist"

    checkpoint_path = os.path.join(
        project_config.CHECKPOINT_PATH,
        f"precision_{precision}",
        gpu_str,
        dist_str,
        training_seed_str,
        unlearning_seed_str,
        "model_checkpoints",
        model_name,
        dataset_name,
        project_config.TIME_NOW,
    )
    fabric.print(f"Checkpoint path: {checkpoint_path}")

    if fabric.global_rank == 0:
        os.makedirs(checkpoint_path, exist_ok=True)
    fabric.barrier()

    # Construct the checkpoint file path template with placeholders
    checkpoint_file_template = os.path.join(
        checkpoint_path, "{model}-{dataset}-{epoch}-{type}.pth"
    )
    fabric.print(f"Checkpoint file template: {checkpoint_file_template}")

    best_acc = 0.0
    weights_path = None
    for epoch in range(1, EPOCHS + 1):
        try:
            fabric.print(f"\n=== Starting Epoch {epoch}/{EPOCHS} ===")

            if epoch > warm:
                if model_name == "ViT":
                    train_scheduler.step()
                else:
                    train_scheduler.step(epoch)

            fabric.print("Starting training phase...")
            train(
                fabric,
                model,
                optimizer,
                train_dataloader,
                loss_function,
                warm,
                warmup_scheduler,
                epoch,
            )

            fabric.print("Starting test phase...")
            acc = eval_training(fabric, model, test_dataloader, loss_function, epoch)

            # Handle checkpointing on rank 0
            fabric.print(f"Testing completed with accuracy: {acc*100:.2f}%")
            if best_acc < acc:
                weights_path = checkpoint_file_template.format(
                    model=model_name,
                    dataset=dataset_name,
                    epoch=epoch,
                    type="best",
                )
                fabric.print(
                    f"New best accuracy! {best_acc*100:.2f}% -> {acc*100:.2f}%"
                )
                fabric.print(f"Saving weights to {weights_path}")
                if distributed_strategy_name.startswith("deepspeed"):
                    # DeepSpeed saves sharded checkpoints by default (directory with
                    # optimizer shards + zero_to_fp32.py). Instead, we extract the full
                    # model state dict so checkpoints are single .pth files compatible
                    # with loading via torch.load() in unlearning/evaluation.
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
                                raw_model = (
                                    model.module if hasattr(model, "module") else model
                                )
                                torch.save(raw_model.state_dict(), weights_path)
                    else:
                        # DS1/DS2: params are full-size on each rank already.
                        if fabric.global_rank == 0:
                            raw_model = (
                                model.module if hasattr(model, "module") else model
                            )
                            torch.save(raw_model.state_dict(), weights_path)
                    fabric.barrier()
                else:
                    # FSDP uses state_dict_type="full" so fabric.save produces a single file.
                    # DDP also works with this format.
                    fabric.save(weights_path, {"model": model})
                best_acc = acc

            fabric.print(f"=== Completed Epoch {epoch}/{EPOCHS} ===\n")

        except Exception as e:
            fabric.print(
                f"Error on rank {fabric.global_rank} in epoch {epoch}: {str(e)}"
            )
            import traceback

            fabric.print(traceback.format_exc())
            raise

    fabric.print(f"Best accuracy achieved: {best_acc:.4f}")

    # Return for cleanup
    return {
        "model": model,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "trainset": trainset,
        "testset": testset,
    }


def main():
    parser = get_common_parser()

    # Add precision argument
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
        "-dataset",
        type=str,
        required=True,
        nargs="?",
        choices=project_config.dataset_names,
        help="dataset to train on",
    )
    parser.add_argument("-classes", type=int, required=True, help="number of classes")
    parser.add_argument(
        "-batch_size", type=int, default=64, help="batch size for dataloader"
    )
    parser.add_argument(
        "-training_seed", type=int, default=None, help="seed for reproducible training"
    )
    parser.add_argument(
        "-unlearning_seed",
        type=int,
        default=None,
        help="seed for reproducible unlearning",
    )
    parser.add_argument(
        "-unlearning_context",
        type=str,
        default="N/A",
        help="A string describing the unlearning context or purpose for this training run.",
    )
    parser.add_argument(
        "-include_gpus_in_path",
        type=str,
        default="true",
        help="Flag to include GPU count in the checkpoint path.",
    )
    args = parser.parse_args()

    model_name = args.net
    dataset_name = args.dataset
    # In SLURM DDP mode with GPU binding:
    # - Each task sees only 1 GPU (CUDA_VISIBLE_DEVICES set per task)
    # - fabric_devices = 1 (what Fabric can actually use per task)
    # - num_gpus = world_size = SLURM_NTASKS (for paths like "4gpus/")
    fabric_devices = (
        1 if MPSAccelerator.is_available() else num_cuda_devices()
    )  # What each task actually sees (1 with GPU binding)
    slurm_ntasks = _os.environ.get("SLURM_NTASKS")
    if slurm_ntasks:
        num_gpus = int(slurm_ntasks)  # World size for paths/logging
    else:
        num_gpus = fabric_devices

    batch_size = args.batch_size
    lr = args.lr

    ##############################################################
    ##############################################################
    precision = args.precision
    class_num = args.classes
    wandb_logging_flag = args.wandb_logging_flag
    tensorboard_logging_flag = args.tensorboard_logging_flag
    csv_logging_flag = args.csv_logging_flag
    logging_root_dir = args.logging_root_dir
    logging_enabled = wandb_logging_flag or tensorboard_logging_flag or csv_logging_flag
    warm = args.warm
    training_seed = args.training_seed  # This can be None if not provided as it is only used for checking if different number of gpus when used return the same results from the same starting weights
    unlearning_seed = (
        args.unlearning_seed
    )  # This cannot be None because we need to use it for unlearning
    use_process_tracker = args.use_process_tracker
    unlearning_context = args.unlearning_context
    include_gpus_in_path = args.include_gpus_in_path == "true"

    MILESTONES = (
        getattr(project_config, f"{dataset_name}_RN_MILESTONES")
        if model_name != "ViT"
        else getattr(project_config, f"{dataset_name}_ViT_MILESTONES")
    )
    EPOCHS = (
        getattr(project_config, f"{dataset_name}_RN_EPOCHS")
        if model_name != "ViT"
        else getattr(project_config, f"{dataset_name}_ViT_EPOCHS")
    )

    # Initialize variables
    tracker = None
    fabric = None
    used_variables = None
    success = True

    try:
        # Conditionally prepare callbacks
        callbacks = []
        if os.getenv("USE_FABRIC_CALLBACKS", "false").lower() == "true":
            callbacks = [
                TrainingCallback(logging_enabled),
                TestCallback(logging_enabled),
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
            "logging_run_name": f"{model_name}_{dataset_name}_precision_{precision}",
        }
        (
            fabric,
            device,
            fabric_strategy,
            use_sync_batchnorm,
            distributed_strategy_name,
        ) = initialize_fabric(fabric_config)

        if use_process_tracker:
            tracker = ProcessTracker(
                fabric=fabric,
                script_type="train",
                model_name=model_name,
                dataset_name=dataset_name,
                num_gpus=num_gpus,
                batch_size=batch_size,  # this does not need scaling because is for each process
            )

        # Initialize WandB
        wandb_project_prefix = os.getenv("WANDB_PROJECT_PREFIX", "R14")
        project_name_parts = [f"{wandb_project_prefix}_TRAINING"]
        project_name_parts.extend(
            [
                model_name,
                dataset_name,
                unlearning_context,
                f"precision_{precision}",
                f"seed_{unlearning_seed}",
                f"{num_gpus}gpus",
                f"dist_{distributed_strategy_name}" if num_gpus > 1 else "no_dist",
            ]
        )
        project_name = "_".join(project_name_parts)
        wandb_config = {
            "wandb_logging_flag": wandb_logging_flag,
            "project_name": project_name,
            "run_name": f"{model_name}_{dataset_name}_classes{class_num}",
            "experiment_config": {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "num_gpus": num_gpus,
                "precision": precision,
                "distributed_strategy": distributed_strategy_name,
                "batch_size": batch_size,
                "training_seed": training_seed,
                "unlearning_seed": unlearning_seed,
                "unlearning_context": unlearning_context,
                "include_gpus_in_path": include_gpus_in_path,
            },
            # Override WandB auto-detected GPU metrics with actual values
            # This ensures WandB reports the GPUs actually used, not all visible GPUs
            # See: https://docs.wandb.ai/ref/python/experiments/settings/
            "actual_gpu_count": num_gpus,
            "actual_gpu_ids": list(range(num_gpus)),
        }

        # Initialize WandB on rank 0 only
        if fabric.global_rank == 0 and wandb_logging_flag:
            fabric = initialize_wandb(fabric, wandb_config)
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
            "num_gpus": num_gpus,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "class_num": class_num,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "warm": warm,
            "MILESTONES": MILESTONES,
            "EPOCHS": EPOCHS,
            "precision": precision,
            "training_seed": training_seed,
            "unlearning_seed": unlearning_seed,
            "unlearning_context": unlearning_context,
            "include_gpus_in_path": include_gpus_in_path,
            "use_sync_batchnorm": use_sync_batchnorm,
            "distributed_strategy_name": distributed_strategy_name,
        }

        if debugger_session:
            with debugger_session:
                used_variables = setup_training(**main_params)
        else:
            used_variables = setup_training(**main_params)

        # Sync WandB results
        if fabric.global_rank == 0 and wandb_logging_flag:
            sync_wandb(fabric)
        fabric.barrier()

    except Exception as e:
        success = False
        handle_distributed_error(fabric, e)

    finally:
        message = (
            "\n================================================================================\n"
            "=== Training completed successfully. Starting cleanup... ===\n"
            "================================================================================\n\n\n"
            if success
            else (
                "\n================================================================================\n"
                "=== Training completed unsuccessfully. Starting cleanup... ===\n"
                "================================================================================\n\n\n"
            )
        )

        if fabric:
            fabric.print(message)
        else:
            print(message)

        # Cleanup to prevent segfaults during distributed exit
        if fabric:
            cleanup(fabric, used_variables)

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
