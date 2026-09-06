import functools
from supreme.eval_metrics.resource_consumption import (
    start_memory_tracking,
    track_memory_usage,
    start_compute_util_tracking,
    track_compute_util_usage,
    start_cpu_util_tracking,
    track_cpu_util_usage,
)
from supreme.utils.memory_utils import memory_usage_in_gb


def resource_log_fields(memory, compute, per_process=False):
    """Flatten available resource fields, tolerating partial/legacy checkpoints.

    Missing measurements stay absent; they are not replaced by invented zeros.
    """
    fields = {}
    sources = (
        (
            memory or {},
            {
                "TotalGPUMemoryGB": "total_gpu_memory",
                "TotalCPUMemoryGB": "total_cpu_memory",
                "MaxGPUMemoryGB": "max_gpu_memory",
                "MaxCPUMemoryGB": "max_cpu_memory",
                "PerProcessGPUMemoryGB": "per_process.gpu_memory",
                "PerProcessCPUMemoryGB": "per_process.cpu_memory",
            },
        ),
        (
            compute or {},
            {
                "GPUIDs": "gpu_ids",
                "LogicalCPUCount": "logical_cpu_count",
                "StartComputeUtilTotal": "start_compute_util.total",
                "StartComputeUtilMax": "start_compute_util.max",
                "EndComputeUtilTotal": "end_compute_util.total",
                "EndComputeUtilMax": "end_compute_util.max",
                "StartComputeUtilPerProcess": "start_compute_util.per_process",
                "EndComputeUtilPerProcess": "end_compute_util.per_process",
                "TotalAverageComputeUtil": "total_avg_compute_util",
                "TotalPeakComputeUtil": "total_peak_compute_util",
                "MaxAverageComputeUtil": "max_avg_compute_util",
                "MaxPeakComputeUtil": "max_peak_compute_util",
                "TotalComputeSeconds": "total_compute_seconds",
                "TotalComputeHours": "total_compute_hours",
                "TotalAverageCPUUtil": "total_avg_cpu_util",
                "TotalPeakCPUUtil": "total_peak_cpu_util",
                "MaxAverageCPUUtil": "max_avg_cpu_util",
                "MaxPeakCPUUtil": "max_peak_cpu_util",
                "TotalCPUSeconds": "total_cpu_seconds",
                "TotalCPUHours": "total_cpu_hours",
                "PerProcessAverageComputeUtil": "per_process.avg_compute_util",
                "PerProcessPeakComputeUtil": "per_process.peak_compute_util",
                "PerProcessComputeSeconds": "per_process.compute_seconds",
                "PerProcessComputeHours": "per_process.compute_hours",
                "PerProcessAverageCPUUtil": "per_process.avg_cpu_util",
                "PerProcessPeakCPUUtil": "per_process.peak_cpu_util",
                "PerProcessCPUSeconds": "per_process.cpu_seconds",
                "PerProcessCPUHours": "per_process.cpu_hours",
            },
        ),
    )
    for source, mapping in sources:
        for label, path in mapping.items():
            if "PerProcess" in label and not per_process:
                continue
            value = source
            for key in path.split("."):
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                fields[label] = value
    return fields


def track_resources(func, *args, **kwargs):
    """
    Track time, memory, and SM utilization of a function execution.
    """

    """
    This function handles two different scenarios for receiving the 'fabric' parameter:

    1. From Decorator (@track_evaluation_metric):
       When used via the @track_evaluation_metric decorator (e.g., in metrics like ZRF),
       'fabric' comes as the first positional argument in args[0]. Example:
       ```python
       @track_evaluation_metric
       def ZRF(fabric, tmodel, retrained_model, forget_dataloader):
           ...
       ```

    2. From Direct Call:
       When called directly (e.g., in unlearn_main.py), 'fabric' is typically included
       in kwargs. Example:
       ```python
       track_resources(
           dynamic_method_call,
           module_path,
           method_name,
           fabric=fabric,  # fabric in kwargs
           **other_kwargs
       )
       ```
    """
    # Try to get fabric from args first, then kwargs
    fabric = args[0] if args else kwargs.get("fabric", None)
    assert fabric is not None, "fabric is None"

    from supreme.eval_metrics.resource_consumption import cancel_resource_tracking

    start_compute_util_data = start_cpu_util_data = None
    try:
        # Start resource tracking
        start_memory_tracking()
        start_compute_util_data = start_compute_util_tracking(fabric)
        start_cpu_util_data = start_cpu_util_tracking(fabric)

        # Execute the function with memory tracking
        peak_mem_usage_gb, result, core_time_dict = memory_usage_in_gb(
            func, *args, **kwargs
        )

        # Track resource usage
        memory_usage_dict = track_memory_usage(fabric, peak_mem_usage_gb)
        local_process_time = core_time_dict["per_process"][fabric.global_rank]
        compute_util_dict = track_compute_util_usage(
            fabric,
            start_compute_util_data,
            local_process_time,
        )
        cpu_util_dict = track_cpu_util_usage(
            fabric,
            start_cpu_util_data,
            local_process_time,
        )
        # Merge CPU util keys into compute_util_dict so downstream consumers
        # (and on-disk JSON) see a single combined resource dict without
        # any signature changes.
        cpu_per_process = cpu_util_dict.get("per_process", {})
        compute_util_dict.setdefault("per_process", {}).update(cpu_per_process)
        compute_util_dict.update(
            {key: value for key, value in cpu_util_dict.items() if key != "per_process"}
        )

        return result, core_time_dict, memory_usage_dict, compute_util_dict
    except BaseException:
        cancel_resource_tracking(start_compute_util_data, start_cpu_util_data)
        raise


class EvaluationMetricTracker:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        # Extract the track_evaluation_resources flag from kwargs if present
        # Default to False - user must explicitly enable evaluation resource tracking
        track_evaluation_resources = kwargs.pop("track_evaluation_resources", False)

        if track_evaluation_resources:
            # Run the evaluation metric with resource tracking
            (
                metric_value_dict,
                core_time_dict,
                memory_usage_dict,
                compute_util_dict,
            ) = track_resources(self.func, *args, **kwargs)

            return {
                "metric_value_dict": metric_value_dict,
                "core_time_dict": core_time_dict,
                "memory_usage_dict": memory_usage_dict,
                "compute_utilisation_dict": compute_util_dict,  # Key kept for compatibility
            }
        else:
            # Run the evaluation metric WITHOUT resource tracking
            metric_value_dict = self.func(*args, **kwargs)

            return {
                "metric_value_dict": metric_value_dict,
                "core_time_dict": None,
                "memory_usage_dict": None,
                "compute_utilisation_dict": None,  # Key kept for compatibility
            }

    def track_epoch_start(self, fabric, epoch, metric_name):
        if fabric.global_rank == 0:
            fabric.call(
                "on_evaluation_epoch_start",
                fabric=fabric,
                epoch=epoch,
                metric_name=metric_name,
            )

    def track_epoch_end(self, fabric, epoch, value=None):
        if fabric.global_rank == 0:
            fabric.call("on_evaluation_epoch_end", epoch=epoch, epoch_value=value)

    def track_batch_start(self, fabric):
        if fabric.global_rank == 0:
            fabric.call("on_evaluation_batch_start")

    def track_batch_end(self, fabric, batch_idx, epoch, value=None):
        if fabric.global_rank == 0:
            fabric.call(
                "on_evaluation_batch_end",
                batch_idx=batch_idx,
                epoch=epoch,
                batch_value=value,
            )


# The decorator is now an instance of this class
track_evaluation_metric = EvaluationMetricTracker
