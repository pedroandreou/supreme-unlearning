import os
import importlib
import torch
from supreme.utils.memory_utils import load_weights_efficiently
import supreme.utils.project_config as project_config
from torch.utils.data import DataLoader


def set_seeds(fabric, seed):
    #########################################################
    # Reproducibility for seeded operations
    # import torch
    # import numpy as np
    # import random

    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    # See Reproducibility Section at:
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    fabric.seed_everything(seed, workers=True)
    #########################################################

    # #########################################################
    # # Full reproducibility for non-deterministic operations

    # # (For CUDA >= 10.2) Set cuBLAS workspace config for deterministic matrix multiplications
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # import torch

    # # See: https://pytorch.org/docs/stable/notes/randomness.html
    # torch.use_deterministic_algorithms(True) # Make PyTorch operations deterministic (where possible)
    # torch.backends.cudnn.deterministic = True # Make cuDNN convolutions deterministic
    # torch.backends.cudnn.benchmark = False # Disable cuDNN benchmarking (for deterministic selection of algorithms)
    # torch.set_deterministic_debug_mode("warn")  # or "error" or "default" # (Optional) Set deterministic debug mode for more control and error reporting
    # #########################################################


def get_root_directory(dataset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if dataset == "PinsFaceRecognition":
        root = os.path.abspath(
            os.path.join(current_dir, "../datasets/data/105_classes_pins_dataset")
        )
    elif dataset == "Caltech101":
        root = os.path.abspath(
            os.path.join(current_dir, "../datasets/data")
        )
    else:
        root = os.path.abspath(os.path.join(current_dir, "../datasets/data/cifar"))
    
    return root


def create_dataloader(
    dataset,
    batch_size,
    is_training=True,
    num_workers=8,
    pin_memory=None,
    num_gpus=1,
    **kwargs,
):
    """
    num_workers is set to 0 because Scalene does not really support multiprocessing, even though it states that it does
    the effective number of workers is 32 if SCALENE is not set in our environment

    Note: In distributed training, each GPU processes batch_size samples, so effective batch size = batch_size * num_gpus.
    To maintain the same effective batch size as single-GPU training, we scale batch_size down by num_gpus.

    See: https://huggingface.co/docs/accelerate/concept_guides/performance
    """

    # pin_memory: only supported on CUDA, not on MPS or CPU
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Scale batch_size for multi-GPU distributed training to maintain same effective batch size
    # effective_batch_size = per_gpu_batch_size * num_gpus
    # To keep effective_batch_size constant, we use: per_gpu_batch_size = batch_size // num_gpus
    if num_gpus > 1:
        scaled_batch_size = batch_size // num_gpus
        if scaled_batch_size < 1:
            scaled_batch_size = 1
        batch_size = scaled_batch_size

    # Create the dataloader with appropriate settings
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_training,  # Only shuffle if training
        num_workers=num_workers if not os.environ.get("SCALENE") else 0,
        pin_memory=pin_memory,
        **kwargs,
        # drop_last=is_training,  # Drop last incomplete batch only during training
    )

    return dataloader


def initialize_network(fabric, model_name, num_labels, device, **kwargs):
    # Define the module path based on the network type
    module_path = f"supreme.models.{model_name}"

    # Check if 'weight_path' is provided in kwargs and load the weights if it is
    weight_path = kwargs.get("weight_path", None)

    # DeepSpeed ZeRO Stage 3 partitions parameters as they are created via
    # deepspeed.zero.Init hooks installed by fabric.init_module(). Those hooks
    # make nn.Module objects non-pickleable, so the rank-0-init + broadcast
    # pattern used for other strategies fails with EOFError inside
    # torch.distributed.broadcast_object_list. For DS3 we initialize on all
    # ranks instead; each rank receives its own shard naturally.
    from lightning.fabric.strategies.deepspeed import DeepSpeedStrategy
    is_ds3 = (
        isinstance(fabric.strategy, DeepSpeedStrategy)
        and getattr(fabric.strategy, "zero_stage_3", False)
    )

    if is_ds3:
        # For DS3 we must NOT use fabric.init_module(): its zero.Init hooks
        # create parameters with shape [0] on each rank, which makes
        # load_state_dict fail with "size mismatch" against full-size
        # checkpoint tensors. Instead, each rank builds the full model
        # independently (same seed -> identical weights), loads its own
        # copy of the checkpoint, and fabric.setup() will shard later when
        # the DeepSpeed engine is created inside the unlearning method.
        net = dynamic_method_call(
            module_name=module_path,
            file_name=model_name,
            num_labels=num_labels,
            fabric=fabric,
        )
        if net:
            total_params = sum(p.numel() for p in net.parameters())
            fabric.print(
                f"[PARAM COUNT] model={model_name} num_labels={num_labels} "
                f"total={total_params} ({total_params/1e6:.2f}M) "
                f"initialised model with {total_params:,} parameters"
            )
        if weight_path and net:
            net = load_weights_efficiently(
                model=net,
                weight_path=weight_path,
                device=device,
            )
            fabric.print(f"Weights loaded from {weight_path}")
        else:
            fabric.print("No weights loaded")
        return net

    net = None
    if fabric.global_rank == 0:
        try:
            # See Efficient initialization at:
            # https://lightning.ai/docs/fabric/2.4.0/advanced/model_init.html
            with fabric.init_module():
                net = dynamic_method_call(
                    module_name=module_path,
                    file_name=model_name,
                    num_labels=num_labels,
                    fabric=fabric,
                )
            if net:
                total_params = sum(p.numel() for p in net.parameters())
                fabric.print(
                    f"[PARAM COUNT] model={model_name} num_labels={num_labels} "
                    f"total={total_params} ({total_params/1e6:.2f}M) "
                    f"initialised model with {total_params:,} parameters"
                )

            # Load weights outside of init_module if provided
            if weight_path and net:
                net = load_weights_efficiently(
                    model=net,
                    weight_path=weight_path,
                    device=device,
                )
                total_params = sum(p.numel() for p in net.parameters())
                fabric.print(
                    f"[PARAM COUNT] model={model_name} num_labels={num_labels} "
                    f"total={total_params} ({total_params/1e6:.2f}M) "
                    f"Weights loaded from {weight_path} with {total_params:,} parameters"
                )
            else:
                fabric.print("No weights loaded")

        except Exception as e:
            fabric.print(f"Error during initialization: {str(e)}")
            raise
    fabric.barrier()
    # Broadcast the model from rank 0 to all other ranks
    net = fabric.broadcast(net, src=0)

    return net


def dynamic_method_call(module_name, file_name, **kwargs):
    """
    This function can be used for loading
    either unlearning methods or models
    """

    fabric = kwargs.get("fabric", None)

    try:
        module = importlib.import_module(module_name)

        # this file would be either a model file or a method file
        # depends from which method is called
        if fabric:
            fabric.print(f"Loading file: '{file_name}' from module: '{module_name}'")
        method = getattr(module, file_name)

        # Remove fabric from kwargs if we're initializing specific models
        if file_name in project_config.model_names:
            kwargs = kwargs.copy()
            kwargs.pop("fabric", None)

        return method(**kwargs)
    except (ModuleNotFoundError, AttributeError) as e:
        if fabric:
            fabric.print(f"Error: {e}")
        else:
            print(f"Error: {e}")
        return None


def strip_per_process_data(d):
    """
    Recursively removes 'per_process' keys from a dictionary or a list of dictionaries.
    This is used to clean up logs when per-process details are not needed.
    """
    if isinstance(d, dict):
        # Create a new dict, excluding keys containing 'per_process' and recursively cleaning other values
        return {
            k: strip_per_process_data(v)
            for k, v in d.items()
            if "per_process" not in k.lower()
        }
    if isinstance(d, list):
        # Recursively clean each item in the list
        return [strip_per_process_data(i) for i in d]
    # Return value as is if it's not a dict or list
    return d
