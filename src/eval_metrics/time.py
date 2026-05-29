from src.utils.unlearning.evaluation_utils import track_evaluation_metric

"""
Time was first introduced as "Timeliness" in the following paper:
Paper: "Towards Making Systems Forget with Machine Unlearning" at https://ieeexplore.ieee.org/document/7163042

Execution Time could be found in the following paper:
Paper: "Selective Synaptic Dampening for Machine Unlearning" at https://arxiv.org/abs/2308.07707
and implementation can be found at: https://github.com/if-loops/selective-synaptic-dampening/blob/cdfdc0e35c1908e032a6e150d882b0fa17833f85/src/forget_full_class_main.py#L201

whereas speedup calculation was implemented based on the metric's description in the first paper mentioned above as no implementation could be found at:
https://github.com/theLauA/MachineUnlearningPy/blob/b59dcd1d6d028b7807a56897b3911fd6989f0c02/lenskit/algorithms/item_knn.py#L141
"""


@track_evaluation_metric
def calculate_time(fabric, retrain_time, unlearn_time):
    """
    Calculate the speedup of unlearning compared to retraining.
    """
    if unlearn_time == 0:
        raise ValueError(
            "Unlearn time cannot be zero, as division by zero is not possible."
        )

    speedup = None
    if retrain_time is not None:
        speedup = retrain_time / unlearn_time
    else:
        # if it is None, then it means it's the retrained model's run so its elapsed time has not been saved yet
        speedup = 1

    return {"final_value": speedup}
