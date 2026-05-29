import functools
from src.eval_metrics.resource_consumption import (
    start_memory_tracking,
    track_memory_usage,
    start_sm_util_tracking,
    track_sm_util_usage,
    start_cpu_util_tracking,
    track_cpu_util_usage,
)
from src.utils.memory_utils import memory_usage_in_gb


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

    # Start resource tracking
    start_memory_tracking()
    start_sm_util_data = start_sm_util_tracking(fabric)
    start_cpu_util_data = start_cpu_util_tracking(fabric)

    # Execute the function with memory tracking
    peak_mem_usage_gb, result, core_time_dict = memory_usage_in_gb(
        func, *args, **kwargs
    )

    # Track resource usage
    memory_usage_dict = track_memory_usage(fabric, peak_mem_usage_gb)
    local_process_time = core_time_dict["per_process"][fabric.global_rank]
    sm_util_dict = track_sm_util_usage(
        fabric,
        start_sm_util_data,
        local_process_time,
    )
    cpu_util_dict = track_cpu_util_usage(
        fabric,
        start_cpu_util_data,
        local_process_time,
    )
    # Merge CPU util keys into sm_util_dict so downstream consumers
    # (and on-disk JSON) see a single combined resource dict without
    # any signature changes.
    sm_util_dict.update(cpu_util_dict)

    return result, core_time_dict, memory_usage_dict, sm_util_dict


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
                sm_util_dict,
            ) = track_resources(self.func, *args, **kwargs)

            return {
                "metric_value_dict": metric_value_dict,
                "core_time_dict": core_time_dict,
                "memory_usage_dict": memory_usage_dict,
                "power_consumption_dict": sm_util_dict,  # Key kept for compatibility
            }
        else:
            # Run the evaluation metric WITHOUT resource tracking
            metric_value_dict = self.func(*args, **kwargs)

            return {
                "metric_value_dict": metric_value_dict,
                "core_time_dict": None,
                "memory_usage_dict": None,
                "power_consumption_dict": None,  # Key kept for compatibility
            }

    def track_epoch_start(self, fabric, epoch, metric_name):
        if fabric.global_rank == 0:
            fabric.call(
                "on_evaluation_epoch_start",
                fabric=fabric,
                epoch=epoch,
                metric_name=metric_name,
            )
        fabric.barrier()

    def track_epoch_end(self, fabric, epoch, value=None):
        if fabric.global_rank == 0:
            fabric.call("on_evaluation_epoch_end", epoch=epoch, epoch_value=value)
        fabric.barrier()

    def track_batch_start(self, fabric):
        if fabric.global_rank == 0:
            fabric.call("on_evaluation_batch_start")
        fabric.barrier()

    def track_batch_end(self, fabric, batch_idx, epoch, value=None):
        if fabric.global_rank == 0:
            fabric.call(
                "on_evaluation_batch_end",
                batch_idx=batch_idx,
                epoch=epoch,
                batch_value=value,
            )
        fabric.barrier()


# The decorator is now an instance of this class
track_evaluation_metric = EvaluationMetricTracker
