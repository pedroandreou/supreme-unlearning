from torch.nn import functional as F
from src.eval_metrics.accuracy import (
    accuracy,
)
import torch
from torch.optim.lr_scheduler import _LRScheduler
import wandb
from src.utils.generic_utils import get_root_directory
import os
from src.utils.debug_utils import (
    export_train_test_data,
    # benchmark_dataloader,
)
import src.datasets.datasets as datasets
from src.utils import project_config
from src.utils.unlearning.evaluation_utils import track_evaluation_metric
from src.utils.generic_utils import create_dataloader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def prepare_dataloaders(
    fabric,
    num_gpus,
    precision,
    model_name,
    seed,
    dataset_name,
    type_of_unlearning_strategy,
    batch_size,
    unlearning,
    export_class_distribution_info_flag,
):
    img_size = 224 if model_name == "ViT" else 32

    if unlearning:
        # For unlearning, derive dataset path from LOG_DIR to maintain consistent path structure
        # (including GPU component like "4gpus/" that run_local.sh sets via LOG_DIR)
        # This ensures trainset.pt/testset.pt are saved in the same location that
        # unlearning_utils.py expects to find them.
        log_dir = os.getenv("LOG_DIR")
        assert log_dir is not None, "LOG_DIR environment variable must be set for unlearning"
        dataset_path = project_config.get_dataset_path_from_log_dir(log_dir, model_name)
    else:
        # Training mode: use centralized path construction
        dataset_path = project_config.get_training_dataset_path(
            precision=precision,
            seed=seed,
            strategy=type_of_unlearning_strategy,
            dataset_name=dataset_name,
            model_name=model_name,
        )

    if fabric.global_rank == 0:
        os.makedirs(dataset_path, exist_ok=True)
    fabric.barrier()

    trainset_full_path = os.path.join(dataset_path, "trainset.pt")
    testset_full_path = os.path.join(dataset_path, "testset.pt")

    # Load datasets if they already exist
    trainset = None
    testset = None
    pin_memory = None

    if fabric.global_rank == 0:
        if all(
            os.path.exists(path) for path in [trainset_full_path, testset_full_path]
        ):
            fabric.print(f"File {trainset_full_path} already exists. Just loading it.")
            trainset = torch.load(str(trainset_full_path), weights_only=False)

            fabric.print(f"File {testset_full_path} already exists. Just loading it.")
            testset = torch.load(str(testset_full_path), weights_only=False)

            pin_memory = False  # data is already on GPU due to broadcasting

        else:
            fabric.print(f"File {trainset_full_path} does not exist. Saving it.")

            root = get_root_directory(dataset_name)

            trainset = getattr(datasets, dataset_name)(
                root=root,
                download=True,
                train=True,
                unlearning=unlearning,
                img_size=img_size,
                model_name=model_name,
            )
            torch.save(trainset, trainset_full_path)

            fabric.print(f"File {testset_full_path} does not exist. Saving it.")
            testset = getattr(datasets, dataset_name)(
                root=root,
                download=True,
                train=False,
                unlearning=unlearning,
                img_size=img_size,
                model_name=model_name,
            )
            torch.save(testset, testset_full_path)

            pin_memory = torch.cuda.is_available()  # pin_memory unsupported on MPS

    fabric.barrier()

    trainset = fabric.broadcast(
        trainset, src=0
    )  # Broadcast trainset from rank 0 to all other ranks

    testset = fabric.broadcast(
        testset, src=0
    )  # Broadcast testset from rank 0 to all other ranks
    pin_memory = fabric.broadcast(pin_memory, src=0)

    # if  fabric.global_rank == 0:
    #     benchmark_dataloader(fabric, num_gpus, trainset, batch_size)
    # fabric.barrier()

    train_dataloader = create_dataloader(
        dataset=trainset,
        batch_size=batch_size,
        is_training=True,
        pin_memory=pin_memory,
        num_gpus=num_gpus,
    )

    test_dataloader = create_dataloader(
        dataset=testset,
        batch_size=batch_size,
        is_training=False,
        pin_memory=pin_memory,
        num_gpus=num_gpus,
    )

    if fabric.global_rank == 0 and export_class_distribution_info_flag:
        try:
            export_train_test_data(
                fabric=fabric,
                seed=seed,
                dataset_name=dataset_name,
                type_of_unlearning_strategy=type_of_unlearning_strategy,
                loader=train_dataloader,
                set_type="train",
                forget_class_name=None,  # this is just the generic dataset
            )
            export_train_test_data(
                fabric=fabric,
                seed=seed,
                dataset_name=dataset_name,
                type_of_unlearning_strategy=type_of_unlearning_strategy,
                loader=test_dataloader,
                set_type="test",
                forget_class_name=None,  # this is just the generic dataset
            )

            fabric.print("Successfully exported dataset information.")
        except Exception as e:
            fabric.print(f"Error exporting dataset information: {str(e)}")
    fabric.barrier()

    return trainset, testset, train_dataloader, test_dataloader


def training_step(model, batch, criterion=F.cross_entropy, neggrad_flag=False):
    images, labels, clabels = batch
    out = model(images)
    loss = -1.0 * criterion(out, clabels) if neggrad_flag else criterion(out, clabels)
    return loss


@track_evaluation_metric
@torch.no_grad()
def evaluate(fabric, model, test_dataloader, epoch=None, do_global_aggregation=False):
    def validation_step(model, batch):
        """
        Paper: "Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher" at https://arxiv.org/abs/2205.08096 uses clabels at GitHub Code: https://github.com/vikram2000b/bad-teaching-unlearning/blob/f1aa988f71cccf1be6d50e0c6f7b2b905e4c9126/utils.py#L18
        &&&
        Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" at https://arxiv.org/abs/2308.07707 uses clabels at GutHub Code: https://github.com/if-loops/selective-synaptic-dampening/blob/cdfdc0e35c1908e032a6e150d882b0fa17833f85/src/utils.py#L31

        WHILE
        Paper: "Zero-Shot Machine Unlearning" at https://arxiv.org/abs/2201.05629 uses just labels at GitHub Code: https://github.com/ayushkumartarun/zero-shot-unlearning/blob/e8881979d20c13280c3ecd351ea1592bdbf62e69/utils.py#L10
        """
        images, labels, clabels = batch

        try:
            out = model(images)
        except Exception as e:
            print(f"Rank {fabric.global_rank}: Exception in model forward: {e}")
            import traceback

            traceback.print_exc()
            raise

        out_cpu = out.detach().cpu()
        clabels_cpu = clabels.detach().cpu()

        loss_cpu = F.cross_entropy(out_cpu, clabels_cpu, reduction="mean")

        acc_val_cpu = accuracy(out_cpu, clabels_cpu)

        if not isinstance(acc_val_cpu, torch.Tensor):
            acc_tensor_cpu = torch.tensor(acc_val_cpu, device="cpu")
        else:
            acc_tensor_cpu = acc_val_cpu.detach().cpu().float()

        return {"Loss": loss_cpu, "Acc": acc_tensor_cpu}

    def validation_epoch_end(fabric, outputs, do_global=False):
        batch_losses_cpu = torch.stack([x["Loss"] for x in outputs])
        batch_accs_cpu = torch.stack([x["Acc"] for x in outputs])

        # Calculate the mean for the current process FIRST
        epoch_loss_cpu = batch_losses_cpu.mean()
        epoch_acc_cpu = batch_accs_cpu.mean()

        if do_global:
            if fabric.world_size > 1:
                # Gather the per-process means
                gathered_losses_device = fabric.all_gather(epoch_loss_cpu)
                gathered_accs_device = fabric.all_gather(epoch_acc_cpu)

                # The final value is the mean of the per-process means
                epoch_loss_device = gathered_losses_device.mean()
                epoch_acc_device = gathered_accs_device.mean()

                epoch_loss = epoch_loss_device.item()
                epoch_acc = epoch_acc_device.item()

            else:
                # In single-process mode, local values are the global values
                epoch_loss = epoch_loss_cpu.item()
                epoch_acc = epoch_acc_cpu.item()
                gathered_losses_device = torch.tensor([epoch_loss])
                gathered_accs_device = torch.tensor([epoch_acc])

            return {
                "Loss": {
                    "final_value": epoch_loss,
                    "per_process": gathered_losses_device.tolist(),
                },
                "Acc": {
                    "final_value": epoch_acc,
                    "per_process": gathered_accs_device.tolist(),
                },
            }

        else:
            epoch_loss = epoch_loss_cpu.item()
            epoch_acc = epoch_acc_cpu.item()

            return {
                "Loss": {"final_value": epoch_loss},
                "Acc": {"final_value": epoch_acc},
            }

    model.eval()
    outputs = []

    # # Track epoch start
    # evaluate.track_epoch_start(fabric, 0, "accuracy")

    for batch_idx, batch_data in enumerate(test_dataloader):
        # evaluate.track_batch_start(fabric)

        images, _, clabels = batch_data

        try:
            # print(f"Rank {fabric.global_rank}: About to run validation_step on batch {batch_idx}")
            result = validation_step(model, batch_data)
            # print(f"Rank {fabric.global_rank}: Finished validation_step on batch {batch_idx}")

            outputs.append(result)

            # evaluate.track_batch_end(fabric, batch_idx, 0, result["Acc"].item())
        except Exception as e:
            fabric.print(
                f"ERROR during validation_step on Rank {fabric.global_rank}, Batch {batch_idx}: {e}"
            )
            import traceback

            traceback.print_exc()
            raise

    if not outputs:
        fabric.print(
            f"SYNC_DEBUG_EVAL_WARNING (Rank {fabric.global_rank}, Epoch {epoch}): No outputs collected in evaluate. Returning NaNs."
        )
        raise ValueError("No outputs collected in evaluate. Returning NaNs.")

    result_dict = validation_epoch_end(fabric, outputs, do_global=do_global_aggregation)

    # # Track epoch end
    # evaluate.track_epoch_end(fabric, 0, result_dict["Acc"]["final_value"])

    if epoch is not None:
        fabric.print(
            f"Epoch {epoch} accuracy: {result_dict['Acc']['final_value']:.4f}, loss: {result_dict['Loss']['final_value']:.4f} ({'local' if not do_global_aggregation else 'global'} result)"
        )
    else:
        fabric.print(
            f"Final accuracy: {result_dict['Acc']['final_value']:.4f}, loss: {result_dict['Loss']['final_value']:.4f} ({'local' if not do_global_aggregation else 'global'} result)"
        )

    return result_dict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit_one_learning_cycle(
    fabric,
    num_gpus,
    epochs,
    model,
    train_dataloader,
    test_dataloader,
    lr=0.01,
    milestones=None,
    wandb_logging_flag=False,
    neggrad_flag=False,
    model_name=None,
):
    history = []

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    is_vit = (model_name == "ViT")

    if is_vit:
        vit_lr = getattr(project_config, "ViT_LR", 5e-5)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=vit_lr,
            weight_decay=0.01,
        )
        fabric.print(f"fit_one_learning_cycle: ViT using AdamW (lr={vit_lr}, wd=0.01)")
    else:
        momentum = 0.9
        weight_decay = 5e-4
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    # Handle both Fabric-wrapped and raw models
    raw_model = model.module if hasattr(model, "module") else model

    # Create ALL schedulers BEFORE fabric.setup() - DeepSpeed wraps the optimizer
    # in FabricDeepSpeedZeroOptimizer which is not recognized by PyTorch schedulers.
    vit_warm = 1
    if is_vit and milestones:
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - vit_warm
        )
        warmup_scheduler = WarmUpLR(optimizer, len(train_dataloader) * vit_warm)
        fabric.print(f"fit_one_learning_cycle: ViT CosineAnnealingLR (T_max={epochs - vit_warm})")
    elif milestones:
        train_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )
        warmup_scheduler = WarmUpLR(optimizer, len(train_dataloader))

    model, optimizer = fabric.setup(raw_model, optimizer)

    if fabric.global_rank == 0 and wandb_logging_flag:
        config_dict = {
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "learning_rate": vit_lr if is_vit else lr,
            "milestones": milestones,
            "neggrad_flag": neggrad_flag,
            "model_name": model_name,
        }
        if not is_vit:
            config_dict["momentum"] = 0.9
            config_dict["weight_decay"] = 5e-4
        wandb.config.update(config_dict)
    fabric.barrier()

    for epoch in range(epochs):
        # Scheduler step: ViT uses cosine after warmup, ResNet uses existing logic
        if is_vit:
            if epoch >= vit_warm and milestones:
                train_scheduler.step()
        else:
            if epoch > 0 and milestones:
                train_scheduler.step()

        model.train()
        train_losses = []
        lrs = []

        for batch_idx, batch in enumerate(train_dataloader):
            loss = training_step(model, batch, criterion, neggrad_flag)
            train_losses.append(loss.detach().cpu())
            fabric.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            current_lr = get_lr(optimizer)
            lrs.append(current_lr)

            # Warmup: ViT warms up for vit_warm epochs, ResNet for 1 epoch (epoch 0 only)
            if is_vit:
                if epoch < vit_warm and milestones:
                    warmup_scheduler.step()
            else:
                if epoch < 1 and milestones:
                    warmup_scheduler.step()

        train_loss = torch.stack(train_losses).mean().item()

        result_dict = evaluate(
            fabric=fabric,
            model=model,
            test_dataloader=test_dataloader,
            epoch=epoch,
            do_global_aggregation=False,
        )
        result = result_dict["metric_value_dict"]

        # Store local results in history
        result["train_loss"] = train_loss
        result["lr"] = lrs[-1]  # Only store the last learning rate
        history.append(result)

    # Return BOTH the wrapped model AND history. Callers must use the returned
    # model (NOT the one they passed in) because fabric.setup() creates a new
    # wrapper. Under FSDP, the new wrapper has valid FlatParameter references
    # while the old one is stale after in-place parameter mutation. Under DDP/
    # DeepSpeed/single-GPU, both wrappers share the same underlying raw model
    # so the old one also works - but callers should be consistent.
    return model, history
