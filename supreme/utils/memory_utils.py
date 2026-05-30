"""
Author: Sebastian Raschka
-------------------------
These functions are from the 'Memory-efficient Model Weight Loading' tutorial
which can be found at: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb
"""

import torch
import os
import psutil
from threading import Thread
import time


def load_weights_efficiently(model, weight_path, device):
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True, mmap=True)
    # Handle both formats: raw state_dict (legacy) and {"model": state_dict} (FSDP/DeepSpeed)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(
        state_dict,
        # assing=True has been commented out even though in the tutorial is used
        # that's because it is used in PyTorch 2.3.0 version - see here: https://pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html
        # where the compatibility matrix of PyTorch 2.3.0 with Fabric lighting is okay, as we can pick 2.3.0 for Pytorch Lighting as well - see: https://lightning.ai/docs/pytorch/stable/versioning.html
        # but then the problem is with torchaudio where the compatibility matrix shows PyTorch 2.1.0 is compatible with TorchAudio 2.1.0 and does not include any new releases (like 2.3.0 that we would expect)
        # see https://pytorch.org/audio/stable/installation.html
        # assign=True, # it performs an in-place operation instead of copying, which can save memory
    )

    return model


def memory_usage_in_gb(func, *args, **kwargs):
    # Get fabric from args
    fabric = args[0] if args else kwargs.get("fabric", None)
    assert fabric is not None, "fabric is None"

    process = psutil.Process(os.getpid())

    # Measure the baseline memory usage before running the function
    baseline_mem = process.memory_info().rss / 1024**3  # in GB

    # Start monitoring memory in a separate thread
    mem_usage = []
    done = False

    def monitor_memory():
        while not done:
            mem_usage.append(process.memory_info().rss / 1024**3)  # Convert to GB
            time.sleep(0.1)

    t = Thread(target=monitor_memory)
    t.start()

    # Run the function and measure its execution time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time

    # Stop monitoring
    done = True
    t.join()

    peak_mem_usage_gb = max(mem_usage) - baseline_mem

    # Gather all process times
    core_time_elapsed_per_process = fabric.all_gather(execution_time)
    
    # Handle single GPU case where all_gather returns a 0-d tensor
    if fabric.world_size == 1:
        # Single GPU: wrap in list to maintain consistent structure
        per_process_times = [core_time_elapsed_per_process.item()]
    else:
        # Multi-GPU: convert tensor to list
        per_process_times = core_time_elapsed_per_process.tolist()
    
    core_time_dict = {
        "final_value": core_time_elapsed_per_process.max().item(),
        "per_process": per_process_times,
    }

    return peak_mem_usage_gb, result, core_time_dict


def cleanup_unlearning_checkpoint(fabric, method_name: str, cleanup_enabled: bool = False):
    """
    Remove the unlearning method model checkpoint after evaluation to save disk space.
    These files can be large (~45MB+ per model) and accumulate quickly across experiments.

    Note: 'retrain' and 'original' methods are NEVER deleted as they serve as reference
    models needed for evaluating other unlearning methods (activation_distance,
    layerwise_distance, completeness, jsdiv, zrf, etc.).

    Args:
        fabric: Lightning Fabric instance for logging
        method_name: Name of the unlearning method (e.g., 'Finetune', 'BadTeacher')
        cleanup_enabled: Whether cleanup is enabled (from command-line flag)
    """
    if not cleanup_enabled:
        return

    # Skip baseline methods that are needed as reference models for evaluation
    protected_methods = {"retrain", "original"}
    if method_name.lower() in protected_methods:
        fabric.print(f"Cleanup: Skipping {method_name} (needed as reference model for evaluation)")
        return

    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        fabric.print("Warning: LOG_DIR not set, skipping checkpoint cleanup")
        return

    # Capitalize method name to match directory structure
    method_capitalized = method_name.capitalize()
    method_dir = os.path.join(log_dir, method_capitalized)
    model_path = os.path.join(method_dir, f"{method_capitalized}_model.pth")

    if os.path.exists(model_path):
        try:
            file_size = os.path.getsize(model_path)
            os.remove(model_path)
            fabric.print(f"Cleanup: Removed {method_capitalized}_model.pth ({file_size / (1024**2):.2f} MB)")
        except Exception as e:
            fabric.print(f"Warning: Could not remove {model_path}: {e}")
    else:
        fabric.print(f"Cleanup: {method_capitalized}_model.pth not found (already removed or doesn't exist)")


def cleanup(fabric, returned_variables=None):
    """
    Proper cleanup to prevent segfaults during distributed exit.
    This is CRITICAL for multi-GPU setups - without explicit cleanup, Python's automatic
    garbage collection causes segfaults when cleaning up Fabric/CUDA objects.

    Args:
        fabric: Lightning Fabric instance
        returned_variables: Dictionary of variables to clean up (optional)
    """
    import gc
    import torch
    import wandb

    try:
        # FIRST: Close WandB if it's active (must be done before other cleanup)
        try:
            if wandb.run is not None:
                fabric.print("Finalizing WandB run...")
                if fabric.global_rank == 0:
                    wandb.finish()
                fabric.barrier()
                fabric.print("WandB finalized")
        except Exception as e:
            fabric.print(f"Warning: WandB finalization issue: {e}")

        # Clear returned variables if they exist
        if returned_variables:
            fabric.print("Cleaning up returned variables...")
            # Explicitly delete each variable to prevent reference cycles
            for key in list(returned_variables.keys()):
                try:
                    returned_variables[key] = None
                except Exception as e:
                    fabric.print(f"Warning: Could not clear {key}: {e}")
            try:
                del returned_variables
            except:
                pass

        # Force garbage collection to cleanup Python objects NOW
        # rather than during interpreter shutdown
        gc.collect()
        fabric.print("Garbage collection complete")

        # Clear GPU cache on all processes
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all CUDA operations to complete
                fabric.print("Cleared CUDA cache and synchronized")
            except Exception as e:
                fabric.print(f"Warning: CUDA cleanup had issues: {e}")
        elif torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
                fabric.print("Cleared MPS cache and synchronized")
            except Exception as e:
                fabric.print(f"Warning: MPS cleanup had issues: {e}")

        # Final barrier before exit to ensure all cleanup is done
        try:
            fabric.barrier()
            fabric.print("✓ Cleanup completed successfully - safe to exit")
        except:
            pass

        # Destroy distributed process group explicitly before interpreter shutdown
        #
        # NOTE: Lightning Fabric normally handles process group cleanup automatically through its
        # teardown mechanism (see Accelerator.teardown() and DDPStrategy.teardown()). However,
        # we've found that explicit destruction here prevents segfaults during Python's final
        # cleanup phase in our specific use case.
        #
        # Why we bypass Fabric's teardown:
        # - Fabric's teardown is designed to be called at specific lifecycle points
        # - Fabric doesn't expose direct access to the underlying torch.distributed process group
        # - We need more control over the timing of process group destruction
        #
        # Implementation: We directly use torch.distributed (from PyTorch, which PyTorch Lightning
        # extends, which Lightning Fabric further extends) to destroy the process group. This is
        # safe because:
        # 1. Fabric uses torch.distributed.ProcessGroup internally (via TorchCollective)
        # 2. dist.destroy_process_group() is idempotent - calling it won't break if Fabric's
        #    teardown has already been invoked
        # 3. Our testing shows this prevents segfaults that occur when relying solely on
        #    Fabric's automatic teardown in multi-GPU training/unlearning scenarios
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                fabric.print("Destroying distributed process group...")
                dist.destroy_process_group()
                fabric.print("Process group destroyed")
        except Exception as e:
            fabric.print(f"Warning: Could not destroy process group: {e}")

    except Exception as e:
        fabric.print(f"⚠ Warning: Cleanup encountered an error: {e}")
        fabric.print("Continuing with exit anyway...")
