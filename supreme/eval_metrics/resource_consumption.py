# Paper: "Learn to Unlearn: Insights Into Machine Unlearning" at https://www.computer.org/csdl/magazine/co/2024/03/10461690/1V5M1o68gKY

# This module tracks resource consumption (memory and GPU compute utilization)
# incurred during the unlearning process, gauging machine unlearning solutions' practical viability and scalability.

# GPU SM UTILIZATION TRACKING:
# We track SM (Streaming Multiprocessor) utilization as the primary GPU compute metric because:
# 1. SM utilization is process-specific - accurately reflects only the workload of our experiment
# 2. SM utilization allows fair comparison across different hardware and usage scenarios
# 3. In multi-user environments (shared compute clusters), per-process metrics are essential
#
# Implementation:
# - NVML mode (required): Uses nvmlDeviceGetProcessUtilization() for per-process SM utilization %
# - Legacy fallback mode: Available for environments without NVML support (less accurate)

import os
import torch
import subprocess
import time
from threading import Thread

# Optional NVML support for per-process GPU metrics
try:
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# Optional psutil support for per-process CPU utilisation
try:
    import psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except Exception:
    _PSUTIL_AVAILABLE = False

# MPS peak memory tracking state (polled in background thread, no hardware counter available)
_mps_peak_memory_gb: float = 0.0
_mps_memory_monitor_flag: bool = False
_mps_memory_monitor_thread = None


# =================================================================== #
# ========================== MEMORY USAGE =========================== #
# =================================================================== #
def start_memory_tracking():
    """Initialize GPU memory tracking."""
    global _mps_peak_memory_gb, _mps_memory_monitor_flag, _mps_memory_monitor_thread
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        # MPS has no hardware peak counter - poll current_allocated_memory() in a thread
        _mps_peak_memory_gb = 0.0
        _mps_memory_monitor_flag = True

        def _poll_mps_memory():
            global _mps_peak_memory_gb, _mps_memory_monitor_flag
            while _mps_memory_monitor_flag:
                current_gb = torch.mps.current_allocated_memory() / (1024**3)
                if current_gb > _mps_peak_memory_gb:
                    _mps_peak_memory_gb = current_gb
                time.sleep(0.1)

        _mps_memory_monitor_thread = Thread(target=_poll_mps_memory, daemon=True)
        _mps_memory_monitor_thread.start()


def track_memory_usage(fabric, peak_memory_used):
    """Track GPU and CPU memory usage across all processes."""
    global _mps_peak_memory_gb, _mps_memory_monitor_flag, _mps_memory_monitor_thread
    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (
            1024**3
        )  # Convert bytes to GB
    elif torch.backends.mps.is_available():
        # Stop polling thread and capture peak
        _mps_memory_monitor_flag = False
        if _mps_memory_monitor_thread is not None:
            _mps_memory_monitor_thread.join(timeout=1.0)
        max_gpu_memory = _mps_peak_memory_gb
    else:
        max_gpu_memory = 0.0

    # Gather memory stats from all processes
    all_gpu_mem = fabric.all_gather(max_gpu_memory)
    all_cpu_mem = fabric.all_gather(peak_memory_used)

    # Handle single GPU case where all_gather returns a 0-d tensor
    # Check if using single GPU (world_size == 1) vs multi-GPU (world_size > 1)
    if fabric.world_size == 1:
        all_gpu_mem = [all_gpu_mem]
        all_cpu_mem = [all_cpu_mem]

    # Ensure tensors are at least 1D before concatenation
    all_gpu_mem = [
        mem.clone().detach().reshape(-1) if mem.dim() == 0 else mem.clone().detach()
        for mem in all_gpu_mem
    ]
    all_cpu_mem = [
        mem.clone().detach().reshape(-1) if mem.dim() == 0 else mem.clone().detach()
        for mem in all_cpu_mem
    ]

    # Calculate both total and maximum memory usage across all processes
    total_gpu_memory = torch.cat(all_gpu_mem).sum().item()
    total_cpu_memory = torch.cat(all_cpu_mem).sum().item()
    max_gpu_memory = torch.cat(all_gpu_mem).max().item()
    max_cpu_memory = torch.cat(all_cpu_mem).max().item()

    memory_usage_dict = {
        "total_gpu_memory": total_gpu_memory,
        "total_cpu_memory": total_cpu_memory,
        "max_gpu_memory": max_gpu_memory,
        "max_cpu_memory": max_cpu_memory,
        "per_process": {
            "gpu_memory": torch.cat(all_gpu_mem).tolist(),
            "cpu_memory": torch.cat(all_cpu_mem).tolist(),
        },
    }

    return memory_usage_dict


# =================================================================== #
# =================================================================== #
# =================================================================== #


# =================================================================== #
# ====================== SM UTILIZATION TRACKING =================== #
# =================================================================== #
compute_utilization_readings = []  # Stores SM utilization percentages
monitor_flag = False  # Global flag for monitoring thread
using_nvml_mode = False  # True: NVML SM-util mode; False: legacy fallback mode


def get_visible_gpu_ids():
    """Get list of GPU IDs, supporting both standalone and SLURM environments."""

    # Try CUDA_VISIBLE_DEVICES first (works on standalone + most SLURM setups)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    if cuda_visible_devices:
        return [int(x.strip()) for x in cuda_visible_devices.split(",") if x.strip()]

    # Fallback: Check SLURM-specific variables
    slurm_gpus = os.environ.get("SLURM_STEP_GPUS") or os.environ.get(
        "GPU_DEVICE_ORDINAL"
    )

    if slurm_gpus:
        return [int(x.strip()) for x in slurm_gpus.split(",") if x.strip()]

    # If nothing is set, check what's available
    import torch

    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    elif torch.backends.mps.is_available():
        return [0]  # MPS is always a single device, index 0
    return []


def _get_mps_gpu_utilization() -> float:
    """Get GPU utilization % on Apple Silicon via ioreg (no sudo required).

    Returns Renderer Utilization % which tracks shader/compute core usage -
    the closest equivalent to NVIDIA SM utilization for ML workloads.
    Falls back to Device Utilization % if Renderer is unavailable.
    """
    try:
        output = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            timeout=2,
        ).decode()
        # Prefer Renderer Utilization (shader/compute cores, most relevant for ML)
        import re

        match = re.search(r'"Renderer Utilization %"\s*=\s*(\d+)', output)
        if not match:
            match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', output)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 0.0


def start_compute_util_tracking(fabric):
    """Initialize GPU SM utilization tracking for the current process.

    Uses NVML for per-process SM utilization when available,
    falls back to legacy mode otherwise.
    """
    global compute_utilization_readings, monitor_flag, using_nvml_mode
    compute_utilization_readings = []
    monitor_flag = True
    using_nvml_mode = False

    # Determine the specific physical GPU ID for this process
    all_physical_ids = get_visible_gpu_ids()
    device_index = fabric.device.index if fabric.device.index is not None else 0
    my_physical_gpu_id = all_physical_ids[device_index] if all_physical_ids else 0

    use_nvml = _NVML_AVAILABLE and os.environ.get("USE_NVML_PER_PROCESS", "0") == "1"
    require_nvml = os.environ.get("REQUIRE_NVML_PER_PROCESS", "0") == "1"

    if use_nvml:
        # Initialize NVML once per process
        # NVML mode provides per-process SM utilization, which is crucial for accurate
        # resource tracking in multi-user environments where multiple processes share GPUs
        try:
            pynvml.nvmlInit()
        except Exception as e:
            if require_nvml:
                raise RuntimeError(
                    f"NVML initialization failed and REQUIRE_NVML_PER_PROCESS=1: {e}"
                )
            # Fallback to basic monitoring if NVML init fails
            use_nvml = False

    if use_nvml:
        using_nvml_mode = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(my_physical_gpu_id)

        # Proactively check per-process utilization support if required.
        # NVMLError_NotFound just means no GPU processes are active yet (idle GPU) -
        # the function is supported, there's simply no data. Only treat other errors
        # (e.g. NVMLError_NotSupported) as "function unavailable".
        if require_nvml:
            try:
                _ = pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
            except pynvml.NVMLError_NotFound:
                pass  # Idle GPU - function is supported, no process records yet
            except Exception as e:
                raise RuntimeError(
                    f"Per-process utilization unsupported but REQUIRE_NVML_PER_PROCESS=1: {e}"
                )

        def monitor_compute_utilization(gpu_id: int, pid: int):
            """Continuously monitor SM utilization for this process using NVML."""
            global monitor_flag, compute_utilization_readings
            handle_local = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            while monitor_flag:
                try:
                    # Get per-process SM utilization (percentage)
                    compute_util_percent = 0.0
                    try:
                        # lastSeenTimeStamp=0 → most recent; sampling ~100ms
                        util_list = pynvml.nvmlDeviceGetProcessUtilization(
                            handle_local, 0
                        )
                        for u in util_list:
                            if getattr(u, "pid", -1) == pid:
                                # Direct SM utilization percentage (0-100)
                                compute_util_percent = float(getattr(u, "smUtil", 0))
                                break
                    except Exception:
                        compute_util_percent = 0.0

                    # Store the SM utilization percentage directly
                    compute_utilization_readings.append(compute_util_percent)

                except Exception:
                    # On any error, append 0 and continue
                    compute_utilization_readings.append(0.0)

                time.sleep(0.1)  # Sample every 100ms

        monitor_thread = Thread(
            target=monitor_compute_utilization,
            args=(my_physical_gpu_id, os.getpid()),
            daemon=True,
        )
        monitor_thread.start()

        # Get initial reading for consistency
        start_value = 0.0
        try:
            util_list = pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
            for u in util_list:
                if getattr(u, "pid", -1) == os.getpid():
                    start_value = float(getattr(u, "smUtil", 0))
                    break
        except Exception:
            start_value = 0.0

        return {
            "start_compute_util": start_value,
            "monitor_thread": monitor_thread,
            "my_gpu_id": my_physical_gpu_id,
            "all_gpu_ids": all_physical_ids,
        }

    # MPS path: sample Renderer Utilization % via ioreg (no sudo required)
    if torch.backends.mps.is_available():

        def monitor_mps():
            global monitor_flag, compute_utilization_readings
            while monitor_flag:
                compute_utilization_readings.append(_get_mps_gpu_utilization())
                time.sleep(0.1)

        monitor_thread = Thread(target=monitor_mps, daemon=True)
        monitor_thread.start()
        start_metric = _get_mps_gpu_utilization()

        return {
            "start_compute_util": start_metric,
            "monitor_thread": monitor_thread,
            "my_gpu_id": my_physical_gpu_id,
            "all_gpu_ids": all_physical_ids,
        }

    # No NVML and no MPS: per-process compute-utilisation tracking is unavailable
    # on this host (e.g. CPU-only). Return a no-op handle - readings stay empty and
    # aggregate to 0. (The legacy nvidia-smi/power.draw fallback has been removed.)
    return {
        "start_compute_util": 0.0,
        "monitor_thread": None,
        "my_gpu_id": my_physical_gpu_id,
        "all_gpu_ids": all_physical_ids,
    }


def track_compute_util_usage(fabric, start_data: dict, process_time: float):
    """Aggregate SM utilization metrics from NVML or fallback mode.

    Returns a dictionary with SM utilization statistics.
    """
    global monitor_flag, compute_utilization_readings, using_nvml_mode

    # Unpack the data collected at the start
    monitor_thread = start_data["monitor_thread"]
    all_gpu_ids = start_data["all_gpu_ids"]

    # Stop monitoring thread if it exists
    if monitor_thread:
        monitor_flag = False
        monitor_thread.join(timeout=1.0)

    if using_nvml_mode:
        # NVML SM-util mode
        if compute_utilization_readings:
            avg_compute_util = sum(compute_utilization_readings) / len(compute_utilization_readings)
            peak_compute_util = max(compute_utilization_readings)
        else:
            avg_compute_util = start_data["start_compute_util"]
            peak_compute_util = start_data["start_compute_util"]

        compute_seconds = float(avg_compute_util) * process_time

        all_start_util = fabric.all_gather(start_data["start_compute_util"])
        all_end_util = fabric.all_gather(avg_compute_util)
        all_avg_util = fabric.all_gather(avg_compute_util)
        all_peak_util = fabric.all_gather(peak_compute_util)
        all_compute_seconds = fabric.all_gather(compute_seconds)

        total_avg_util = all_avg_util.mean().item()
        total_peak_util = all_peak_util.max().item()
        max_avg_util = all_avg_util.max().item()
        max_peak_util = all_peak_util.max().item()
        total_compute_seconds = all_compute_seconds.sum().item()

        compute_util_dict = {
            "gpu_ids": ",".join(map(str, all_gpu_ids)),
            "start_compute_util": {
                "total": all_start_util.sum().item(),
                "max": all_start_util.max().item(),
                "per_process": all_start_util.tolist(),
            },
            "end_compute_util": {
                "total": all_end_util.sum().item(),
                "max": all_end_util.max().item(),
                "per_process": all_end_util.tolist(),
            },
            "total_avg_compute_util": total_avg_util,
            "total_peak_compute_util": total_peak_util,
            "max_avg_compute_util": max_avg_util,
            "max_peak_compute_util": max_peak_util,
            "total_compute_seconds": total_compute_seconds,
            "total_compute_hours": total_compute_seconds / 3600,
            "per_process": {
                "avg_compute_util": all_avg_util.tolist(),
                "peak_compute_util": all_peak_util.tolist(),
                "compute_seconds": all_compute_seconds.tolist(),
                "compute_hours": (all_compute_seconds / 3600).tolist(),
            },
        }
    else:
        # MPS (Apple Silicon) sampling mode, or no-tracking mode (no NVML/MPS).
        # Aggregate the sampled readings; if none were collected, fall back to the
        # start value. No legacy nvidia-smi/power.draw path.
        if compute_utilization_readings:
            avg_metric = sum(compute_utilization_readings) / len(compute_utilization_readings)
            peak_metric = max(compute_utilization_readings)
        else:
            avg_metric = start_data["start_compute_util"]
            peak_metric = start_data["start_compute_util"]

        derived_seconds = float(avg_metric) * process_time
        derived_hours = derived_seconds / 3600

        all_start_metric = fabric.all_gather(start_data["start_compute_util"])
        all_end_metric = fabric.all_gather(avg_metric)
        all_avg_metric = fabric.all_gather(avg_metric)
        all_peak_metric = fabric.all_gather(peak_metric)
        all_derived_seconds = fabric.all_gather(derived_seconds)
        all_derived_hours = fabric.all_gather(derived_hours)

        total_avg_metric = all_avg_metric.sum().item()
        total_peak_metric = all_peak_metric.sum().item()
        max_avg_metric = all_avg_metric.max().item()
        max_peak_metric = all_peak_metric.max().item()
        total_derived_seconds = all_derived_seconds.sum().item()
        total_derived_hours = all_derived_hours.sum().item()

        compute_util_dict = {
            "gpu_ids": ",".join(map(str, all_gpu_ids)),
            "start_compute_util": {
                "total": all_start_metric.sum().item(),
                "max": all_start_metric.max().item(),
                "per_process": all_start_metric.tolist(),
            },
            "end_compute_util": {
                "total": all_end_metric.sum().item(),
                "max": all_end_metric.max().item(),
                "per_process": all_end_metric.tolist(),
            },
            "total_avg_compute_util": total_avg_metric,
            "total_peak_compute_util": total_peak_metric,
            "max_avg_compute_util": max_avg_metric,
            "max_peak_compute_util": max_peak_metric,
            "total_compute_seconds": total_derived_seconds,
            "total_compute_hours": total_derived_hours,
            "per_process": {
                "avg_compute_util": all_avg_metric.tolist(),
                "peak_compute_util": all_peak_metric.tolist(),
                "compute_seconds": all_derived_seconds.tolist(),
                "compute_hours": all_derived_hours.tolist(),
            },
        }

    return compute_util_dict


# =================================================================== #
# =================================================================== #
# =================================================================== #


# =================================================================== #
# ====================== CPU UTILIZATION TRACKING ================== #
# =================================================================== #
# Per-process CPU utilisation, sampled via psutil.Process.cpu_percent().
# Hardware-agnostic (Linux/macOS/Windows). Values are normalised by the
# logical CPU count so they stay within [0, 100] and are comparable
# across machines with different core counts. The raw logical core count
# is also returned so the un-normalised value can be reconstructed.
#
# Notes for consumers loading old artefacts: JSON dicts saved before this
# metric existed will lack the CPU util keys - this is intentional
# ("code only, no re-run") and downstream code should use .get() if it
# needs to tolerate old files.

cpu_util_readings = []  # Normalised per-process CPU % (0-100)
cpu_monitor_flag = False
cpu_monitor_thread = None


def start_cpu_util_tracking(fabric):
    """Initialise per-process CPU utilisation tracking for the current process.

    Spawns a daemon thread that polls psutil.Process.cpu_percent() at 10 Hz
    and appends normalised readings to cpu_util_readings.
    """
    global cpu_util_readings, cpu_monitor_flag, cpu_monitor_thread
    cpu_util_readings = []
    cpu_monitor_flag = True
    cpu_monitor_thread = None

    logical_cpus = 1
    if _PSUTIL_AVAILABLE:
        logical_cpus = max(psutil.cpu_count(logical=True) or 1, 1)

    if not _PSUTIL_AVAILABLE:
        # Graceful degradation: no thread, empty readings.
        return {
            "monitor_thread": None,
            "start_cpu_util": 0.0,
            "logical_cpu_count": logical_cpus,
        }

    # Instantiate Process once and reuse inside the thread. Re-constructing
    # it per iteration would reset psutil's internal timestamp and every
    # sample would read 0.0.
    proc = psutil.Process(os.getpid())

    # Prime the baseline: the first call to cpu_percent(interval=None)
    # always returns 0.0 because psutil needs two observations to compute
    # a delta. Discard it so the first real sample is meaningful.
    try:
        proc.cpu_percent(interval=None)
    except Exception:
        pass

    def monitor_cpu_utilization():
        global cpu_monitor_flag, cpu_util_readings
        while cpu_monitor_flag:
            try:
                raw = proc.cpu_percent(interval=None)
                normalised = raw / logical_cpus
                cpu_util_readings.append(normalised)
            except psutil.NoSuchProcess:
                cpu_util_readings.append(0.0)
            except Exception:
                cpu_util_readings.append(0.0)
            time.sleep(0.1)  # Sample every 100ms

    cpu_monitor_thread = Thread(target=monitor_cpu_utilization, daemon=True)
    cpu_monitor_thread.start()

    return {
        "monitor_thread": cpu_monitor_thread,
        "start_cpu_util": 0.0,
        "logical_cpu_count": logical_cpus,
    }


def track_cpu_util_usage(fabric, start_data: dict, process_time: float):
    """Aggregate CPU utilisation metrics across all ranks.

    Mirrors track_compute_util_usage: stops the monitor thread, computes local
    avg/peak, then all_gathers across ranks to produce a dict of the same
    shape as the SM util dict.
    """
    global cpu_monitor_flag, cpu_util_readings

    monitor_thread = start_data.get("monitor_thread")
    logical_cpus = start_data.get("logical_cpu_count", 1)

    if monitor_thread is not None:
        cpu_monitor_flag = False
        monitor_thread.join(timeout=1.0)

    if cpu_util_readings:
        avg_cpu_util = sum(cpu_util_readings) / len(cpu_util_readings)
        peak_cpu_util = max(cpu_util_readings)
    else:
        avg_cpu_util = 0.0
        peak_cpu_util = 0.0

    cpu_seconds = float(avg_cpu_util) * process_time

    all_avg_util = fabric.all_gather(avg_cpu_util)
    all_peak_util = fabric.all_gather(peak_cpu_util)
    all_cpu_seconds = fabric.all_gather(cpu_seconds)

    total_avg_util = all_avg_util.mean().item()
    total_peak_util = all_peak_util.max().item()
    max_avg_util = all_avg_util.max().item()
    max_peak_util = all_peak_util.max().item()
    total_cpu_seconds = all_cpu_seconds.sum().item()

    return {
        "logical_cpu_count": logical_cpus,
        "total_avg_cpu_util": total_avg_util,
        "total_peak_cpu_util": total_peak_util,
        "max_avg_cpu_util": max_avg_util,
        "max_peak_cpu_util": max_peak_util,
        "total_cpu_seconds": total_cpu_seconds,
        "total_cpu_hours": total_cpu_seconds / 3600,
        "per_process": {
            "avg_cpu_util": all_avg_util.tolist(),
            "peak_cpu_util": all_peak_util.tolist(),
            "cpu_seconds": all_cpu_seconds.tolist(),
            "cpu_hours": (all_cpu_seconds / 3600).tolist(),
        },
    }


# =================================================================== #
# =================================================================== #
# =================================================================== #


# =================================================================== #
# ========================== STORAGE COSTS ========================== #
# =================================================================== #

# =================================================================== #
# =================================================================== #
# =================================================================== #
