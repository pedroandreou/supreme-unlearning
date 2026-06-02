import os
from supreme.utils.memory_utils import load_weights_efficiently
import torch
import json
from typing import Optional, Tuple, Dict, Any


def initialize_paths(log_dir: str, method_name: str) -> Dict[str, str]:
    """Initialize all paths based on the provided log directory and method name."""
    method_dir = os.path.join(log_dir, method_name)

    return {
        "method_dir": method_dir,
        "model_path": f"{method_dir}/{method_name}_model.pth",
        "time_elapsed_path": f"{method_dir}/{method_name}_time_elapsed.json",
        "memory_path": f"{method_dir}/{method_name}_memory_usage.json",
        "power_path": f"{method_dir}/{method_name}_compute_utilisation.json",
    }


def get_model_save_path(method_name: str) -> str:
    """Return the path where the unlearning model checkpoint should be saved.

    This is used when saving via fabric.save() (for DDP/FSDP) where the model
    save is done separately from the log saves. The directory is created if needed.
    """
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is not set.")
    paths = initialize_paths(log_dir, method_name)
    os.makedirs(paths["method_dir"], exist_ok=True)
    return paths["model_path"]


def save_logs_only(
    fabric,
    method_name,
    core_time_dict: Optional[Dict] = None,
    memory_usage_dict: Optional[Dict] = None,
    compute_utilisation_dict: Optional[Dict] = None,
):
    """Save only the time/memory/power log files (not the model).

    Used when the model was already saved via fabric.save() in the DDP/FSDP path.
    """
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is not set.")
    paths = initialize_paths(log_dir, method_name)
    os.makedirs(paths["method_dir"], exist_ok=True)

    if all(
        x is not None
        for x in [core_time_dict, memory_usage_dict, compute_utilisation_dict]
    ):
        with open(paths["time_elapsed_path"], "w") as f:
            json.dump(core_time_dict, f, indent=4)
        fabric.print(
            f'{method_name} core time data saved to {paths["time_elapsed_path"]}'
        )

        with open(paths["memory_path"], "w") as f:
            json.dump(memory_usage_dict, f, indent=4)
        fabric.print(f'{method_name} memory usage saved to {paths["memory_path"]}')

        with open(paths["power_path"], "w") as f:
            json.dump(compute_utilisation_dict, f, indent=4)
        fabric.print(f'{method_name} power consumption saved to {paths["power_path"]}')


def save_model_and_logs(
    fabric,
    model=None,
    method_name=None,
    core_time_dict: Optional[Dict] = None,  # Optional for unlearning_teacher
    memory_usage_dict: Optional[Dict] = None,  # Optional for unlearning_teacher
    compute_utilisation_dict: Optional[Dict] = None,  # Optional for unlearning_teacher
    state_dict=None,  # Pre-extracted state_dict (needed for FSDP where all ranks must participate in state_dict())
):
    """
    Save the model and optionally its logs to the log directory.
    For unlearning_teacher, only saves the model state.
    For unlearned models, saves model state and all logs.

    Args:
        state_dict: If provided, saves this directly instead of calling model.state_dict().
            This is needed for FSDP: extracting the state_dict requires all ranks to participate
            in an all-gather, so it must be done BEFORE entering a rank-0-only code block.
    """
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is not set.")

    # Initialize all paths
    paths = initialize_paths(log_dir, method_name)

    # Create a subdirectory for the method_name inside the log_dir
    os.makedirs(paths["method_dir"], exist_ok=True)

    # Save the model state dict
    if state_dict is None:
        state_dict = model.state_dict()
    torch.save(state_dict, paths["model_path"])
    fabric.print(f'{method_name} model saved to {paths["model_path"]}')

    # Only save logs if they are provided (for unlearned models)
    if all(
        x is not None
        for x in [core_time_dict, memory_usage_dict, compute_utilisation_dict]
    ):
        with open(paths["time_elapsed_path"], "w") as f:
            json.dump(core_time_dict, f, indent=4)
        fabric.print(
            f'{method_name} core time data saved to {paths["time_elapsed_path"]}'
        )

        with open(paths["memory_path"], "w") as f:
            json.dump(memory_usage_dict, f, indent=4)
        fabric.print(f'{method_name} memory usage saved to {paths["memory_path"]}')

        with open(paths["power_path"], "w") as f:
            json.dump(compute_utilisation_dict, f, indent=4)
        fabric.print(f'{method_name} power consumption saved to {paths["power_path"]}')


def load_model_and_logs(
    fabric, net, method_name, device
) -> Tuple[Any, Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load the model and optionally its logs from the log directory.
    For unlearning_teacher, only loads and returns the model.
    For unlearned models, loads and returns model and all logs.
    """
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is not set.")

    # Initialize all paths
    paths = initialize_paths(log_dir, method_name)

    # Load the model weights
    if os.path.exists(paths["model_path"]):
        fabric.print(f'Loading "{method_name}" model from "{paths["model_path"]}"')
        net = load_weights_efficiently(
            model=net, weight_path=paths["model_path"], device=device
        )
    else:
        raise FileNotFoundError(
            f'Model for {method_name} not found at {paths["model_path"]}'
        )

    # Check if this is unlearning_teacher (which doesn't have logs)
    if method_name == "Unlearning_teacher":
        return net, None, None, None

    # For unlearned models, load all logs
    with open(paths["time_elapsed_path"], "r") as f:
        core_time_dict = json.load(f)

    with open(paths["memory_path"], "r") as f:
        memory_usage_dict = json.load(f)

    with open(paths["power_path"], "r") as f:
        compute_utilisation_dict = json.load(f)

    return net, core_time_dict, memory_usage_dict, compute_utilisation_dict


def check_model_files_exist(
    fabric,
    model: Any,
    method_name: str,
    device: Any,
) -> Tuple[Any, bool, Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Checks if model and log files exist for a given method.
    For unlearning_teacher, only checks model file.
    For other methods, checks model and all log files.

    Returns:
        model: The model (loaded if files exist)
        files_exist: Boolean indicating if files were found
        core_time_dict: Time data as a dictionary (if files exist and not unlearning_teacher)
        memory_usage_dict: Memory usage data (if files exist and not unlearning_teacher)
        compute_utilisation_dict: Power consumption data (if files exist and not unlearning_teacher)
    """
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is not set.")

    # Initialize all paths
    paths = initialize_paths(log_dir, method_name)

    # For unlearning_teacher, only check if model exists
    if method_name == "Unlearning_teacher":
        files_exist = os.path.exists(paths["model_path"])
    else:
        files_exist = (
            os.path.exists(paths["model_path"])
            and os.path.exists(paths["time_elapsed_path"])
            and os.path.exists(paths["memory_path"])
            and os.path.exists(paths["power_path"])
        )

    fabric.print(
        "About to check if files exist condition in check_model_files_exist func"
    )

    core_time_dict = None
    memory_usage_dict = None
    compute_utilisation_dict = None

    if files_exist:
        if method_name == "Unlearning_teacher":
            fabric.print(f"The '{method_name}' model exists. Loading ...")
        else:
            fabric.print(
                f"The '{method_name}' model and its time elapsed, memory usage and power consumption exist. Loading ..."
            )

        (
            model,
            core_time_dict,
            memory_usage_dict,
            compute_utilisation_dict,
        ) = load_model_and_logs(
            fabric=fabric,
            net=model,
            method_name=method_name,
            device=device,
        )
    else:
        if method_name == "Unlearning_teacher":
            fabric.print(f"The '{method_name}' model does not exist. Initializing ...")
        else:
            fabric.print(
                f"The '{method_name}' model and its time elapsed do not exist. Training and then saving ..."
            )

    return (
        model,
        files_exist,
        core_time_dict,
        memory_usage_dict,
        compute_utilisation_dict,
    )


def save_evaluation_results(fabric, method_name: str, eval_result: Dict) -> None:
    """
    Save evaluation metrics to a JSON file in the method's log directory.
    If the file already exists, merge new metrics into it (append mode).

    Args:
        fabric: Lightning Fabric instance for printing
        method_name: Name of the unlearning method (e.g., "Finetune")
        eval_result: Dictionary of evaluation metrics to save
    """
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is not set.")

    paths = initialize_paths(log_dir, method_name)
    eval_path = os.path.join(paths["method_dir"], f"{method_name}_eval_results.json")

    # If file exists, load and merge (append new metrics without overwriting existing ones)
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            existing = json.load(f)
        # Deep merge: update top-level keys from eval_result into existing
        _deep_merge(existing, eval_result)
        merged = existing
        fabric.print(f"Appending new evaluation metrics to existing {eval_path}")
    else:
        os.makedirs(paths["method_dir"], exist_ok=True)
        merged = eval_result
        fabric.print(f"Saving evaluation metrics to {eval_path}")

    with open(eval_path, "w") as f:
        json.dump(merged, f, indent=4)


def _deep_merge(base: dict, override: dict) -> None:
    """
    Recursively merge override into base in-place.
    New keys from override are added; existing leaf values are not overwritten.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key not in base:
            base[key] = value
