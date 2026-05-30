# metrics that do not require retrained model
from supreme.utils.training.training_utils import evaluate
from supreme.eval_metrics.zrf import ZRF

# metrics that require retrained model
from supreme.eval_metrics.activation_distance import (
    actv_dist,
)
from supreme.eval_metrics.membership_inference_attack import (
    get_membership_attack_prob,
)
from supreme.eval_metrics.layerwise_distance import (
    lay_dist,
)
from supreme.eval_metrics.completeness import (
    calculate_completeness,
)
from supreme.eval_metrics.time import (
    calculate_time,
)
from lightning.fabric import Fabric
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def _is_fsdp_wrapped(model):
    """Return True if any submodule (or the model itself) is wrapped in FSDP.

    Used for layerwise_distance: when a model is FSDP-wrapped, its parameters
    are sharded across ranks, so element-wise comparison of local shards would
    give wrong results. We must gather full params via summon_full_params()
    before iterating named_parameters().
    """
    return any(isinstance(m, FSDP) for m in model.modules())


# Returns metrics
def get_metric_scores(
    fabric: Fabric,
    num_gpus: int,
    eval_metrics: list,
    lr: float,
    batch_size: int,
    original_model,
    unlearned_model,
    unlearning_teacher,
    retrained_model,
    retain_train_dataloader,
    retain_test_dataloader,
    forget_train_dataloader,
    forget_test_dataloader,
    train_dataloader,
    test_dataloader,
    trainset,
    core_time_dict,
    memory_usage_dict,
    power_consumption_dict,
    wandb_logging_flag,
    retrain_time_elapsed_dict,
    track_evaluation_resources=False,
    **kwargs,
):
    # If it's retrained model's run, then it hasn't been saved yet, so it's None and the unlearned_model variable is the retrained model
    # so we compare it with itself in the following metrics in the rest of the Python file as no unlearning method is applied to it
    reference_model = (
        retrained_model if retrained_model is not None else unlearned_model
    )

    # If world size is 1, then we do not need to gather the results
    # Some metrics like accuracy have do_global_aggregation=True as they are called by other metrics like RT, AIN, RAIN as do_global_aggregation=False
    # and therefore they have the check of fabric.world_size > 1 to not gather the results inside of them
    # So global for them is considered here as it is called just for it and not from other metrics
    do_global_aggregation = True if fabric.world_size > 1 else False

    unlearning_method_resources = {
        "core_time_elapsed": core_time_dict,
        "memory_usage": memory_usage_dict,
        "power_consumption": power_consumption_dict,
    }

    """
    If any of the metrics below in the "result" dictionary do not have a "test" key, then it means that the metric is not dependent on the type of the dataset
    The "test" key is used for potential future extension to a test set
    """
    result = {}
    ################################# Accuracy #################################
    if "accuracy" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating Accuracy and Loss", end=" ")

        fabric.print("\nCalculating Accuracy and Loss on the whole Test Set")
        # Test Set
        test_acc_loss_dict = evaluate(
            fabric=fabric,
            model=unlearned_model,
            test_dataloader=test_dataloader,
            do_global_aggregation=True,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print(
            "The accuracy and loss on the whole test set is: ",
            "Acc: ",
            test_acc_loss_dict["metric_value_dict"]["Acc"]["final_value"],
            "Loss: ",
            test_acc_loss_dict["metric_value_dict"]["Loss"]["final_value"],
        )

        fabric.print("\nCalculating Accuracy and Loss on the Retain Test Set", end=" ")
        test_retain_acc_loss_dict = evaluate(
            fabric=fabric,
            model=unlearned_model,
            test_dataloader=retain_test_dataloader,
            do_global_aggregation=True,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print(
            "The accuracy and loss on the retain test set is: ",
            "Acc: ",
            test_retain_acc_loss_dict["metric_value_dict"]["Acc"]["final_value"],
            "Loss: ",
            test_retain_acc_loss_dict["metric_value_dict"]["Loss"]["final_value"],
        )

        fabric.print("\nCalculating Accuracy and Loss on the Forget Test Set", end=" ")
        test_forget_acc_loss_dict = evaluate(
            fabric=fabric,
            model=unlearned_model,
            test_dataloader=forget_test_dataloader,
            do_global_aggregation=True,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print(
            "The accuracy and loss on the forget test set is: ",
            "Acc: ",
            test_forget_acc_loss_dict["metric_value_dict"]["Acc"]["final_value"],
            "Loss: ",
            test_forget_acc_loss_dict["metric_value_dict"]["Loss"]["final_value"],
        )

        result["accuracy"] = {
            "test": {
                "unlearning_method": {
                    # Whole Test Set
                    "whole_acc": test_acc_loss_dict["metric_value_dict"]["Acc"],
                    # Retain Test Set
                    "retain_acc": test_retain_acc_loss_dict["metric_value_dict"]["Acc"],
                    # Forget Test Set
                    "forget_acc": test_forget_acc_loss_dict["metric_value_dict"]["Acc"],
                },
                "eval_metric": {
                    "core_time_elapsed": test_acc_loss_dict[
                        "core_time_dict"
                    ],  # we let it be the same with "Loss"
                    "memory_usage": test_acc_loss_dict[
                        "memory_usage_dict"
                    ],  # we let it be the same with "Loss"
                    "power_consumption": test_acc_loss_dict[
                        "power_consumption_dict"
                    ],  # we let it be the same with "Loss"
                },
            },
        }
        result["accuracy"]["test"]["unlearning_method"].update(
            unlearning_method_resources
        )

        result["loss"] = {
            "test": {
                "unlearning_method": {
                    # Whole Test Set
                    "whole_loss": test_acc_loss_dict["metric_value_dict"]["Loss"],
                    # Retain Test Set
                    "retain_loss": test_retain_acc_loss_dict["metric_value_dict"][
                        "Loss"
                    ],
                    # Forget Test Set
                    "forget_loss": test_forget_acc_loss_dict["metric_value_dict"][
                        "Loss"
                    ],
                },
                "eval_metric": {
                    "core_time_elapsed": test_acc_loss_dict[
                        "core_time_dict"
                    ],  # we let it be the same with "Accuracy"
                    "memory_usage": test_acc_loss_dict[
                        "memory_usage_dict"
                    ],  # we let it be the same with "Accuracy"
                    "power_consumption": test_acc_loss_dict[
                        "power_consumption_dict"
                    ],  # we let it be the same with "Accuracy"
                },
            },
        }
        result["loss"]["test"]["unlearning_method"].update(unlearning_method_resources)
    #############################################################################

    # ----

    ################################# ZRF ######################################
    if "zrf" in eval_metrics:
        fabric.print("================================================")
        # Test Set
        fabric.print("\nCalculating Initial ZRF")
        test_initial_zrf_dict = ZRF(
            fabric=fabric,
            model1=original_model,
            model2=unlearning_teacher,
            test_dataloader=forget_test_dataloader,
            metric_name="initial_zrf",
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )  # Initial Unlearning Score
        fabric.print("Finished Calculating Initial ZRF")
        fabric.print(
            "The initial ZRF value is: ",
            test_initial_zrf_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("\n================================================")
        fabric.print("\nCalculating Final ZRF")
        test_final_zrf_dict = ZRF(
            fabric=fabric,
            model1=unlearned_model,
            model2=unlearning_teacher,
            test_dataloader=forget_test_dataloader,
            metric_name="final_zrf",
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )  # Final Unlearning Score where the unlearned_model is student_model for the bad_teacher unlearning method
        fabric.print("Finished Calculating Final ZRF")
        fabric.print(
            "The final ZRF value is: ",
            test_final_zrf_dict["metric_value_dict"]["final_value"],
        )

        result["zrf"] = {
            "test": {
                "unlearning_method": {
                    # Initial ZRF
                    "initial_zrf": test_initial_zrf_dict["metric_value_dict"],
                    # Final ZRF
                    "final_zrf": test_final_zrf_dict["metric_value_dict"],
                },
                "eval_metric": {
                    "initial_zrf": {
                        "core_time_elapsed": test_initial_zrf_dict["core_time_dict"],
                        "memory_usage": test_initial_zrf_dict["memory_usage_dict"],
                        "power_consumption": test_initial_zrf_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "final_zrf": {
                        "core_time_elapsed": test_final_zrf_dict["core_time_dict"],
                        "memory_usage": test_final_zrf_dict["memory_usage_dict"],
                        "power_consumption": test_final_zrf_dict[
                            "power_consumption_dict"
                        ],
                    },
                },
            },
        }
        result["zrf"]["test"]["unlearning_method"].update(unlearning_method_resources)
    ############################################################################

    # ----

    # ############################ JS-Divergence ##################################
    if "jsdiv" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating JS-Divergence", end=" ")

        # Whole Test Set
        fabric.print("\nCalculating JS-Divergence on Whole Set", end=" ")
        test_jsdiv_whole_dict = ZRF(
            fabric=fabric,
            model1=reference_model,
            model2=unlearned_model,
            test_dataloader=test_dataloader,
            metric_name="jsdiv",
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        test_jsdiv_whole_dict["metric_value_dict"]["final_value"] = (
            1 - test_jsdiv_whole_dict["metric_value_dict"]["final_value"]
        )
        fabric.print("Finished Calculating JS-Divergence on Whole Set")
        fabric.print(
            "The JS-Divergence (Whole Set) value is: ",
            test_jsdiv_whole_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("================================================")
        fabric.print("\nCalculating JS-Divergence on Retain Set", end=" ")
        # Retain Test Set
        test_jsdiv_retain_dict = ZRF(
            fabric=fabric,
            model1=reference_model,
            model2=unlearned_model,
            test_dataloader=retain_test_dataloader,
            metric_name="jsdiv",
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        test_jsdiv_retain_dict["metric_value_dict"]["final_value"] = (
            1 - test_jsdiv_retain_dict["metric_value_dict"]["final_value"]
        )
        fabric.print("Finished Calculating JS-Divergence on Retain Set")
        fabric.print(
            "The JS-Divergence (Retain Set) value is: ",
            test_jsdiv_retain_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("================================================")
        fabric.print("\nCalculating JS-Divergence on Forget Set", end=" ")
        # Forget Test Set
        test_jsdiv_forget_dict = ZRF(
            fabric=fabric,
            model1=reference_model,
            model2=unlearned_model,
            test_dataloader=forget_test_dataloader,
            metric_name="jsdiv",
            do_global_aggregation=False,
            track_evaluation_resources=track_evaluation_resources,
        )
        test_jsdiv_forget_dict["metric_value_dict"]["final_value"] = (
            1 - test_jsdiv_forget_dict["metric_value_dict"]["final_value"]
        )
        fabric.print("Finished Calculating JS-Divergence on Forget Set")
        fabric.print(
            "The JS-Divergence (Forget Set) value is: ",
            test_jsdiv_forget_dict["metric_value_dict"]["final_value"],
        )

        result["jsdiv"] = {
            "test": {
                "unlearning_method": {
                    # Whole Test Set
                    "jsdiv_whole": test_jsdiv_whole_dict["metric_value_dict"],
                    # Retain Test Set
                    "jsdiv_retain": test_jsdiv_retain_dict["metric_value_dict"],
                    # Forget Test Set
                    "jsdiv_forget": test_jsdiv_forget_dict["metric_value_dict"],
                },
                "eval_metric": {
                    "jsdiv_whole": {
                        "core_time_elapsed": test_jsdiv_whole_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": test_jsdiv_whole_dict["memory_usage_dict"],
                        "power_consumption": test_jsdiv_whole_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "jsdiv_retain": {
                        "core_time_elapsed": test_jsdiv_retain_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": test_jsdiv_retain_dict["memory_usage_dict"],
                        "power_consumption": test_jsdiv_retain_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "jsdiv_forget": {
                        "core_time_elapsed": test_jsdiv_forget_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": test_jsdiv_forget_dict["memory_usage_dict"],
                        "power_consumption": test_jsdiv_forget_dict[
                            "power_consumption_dict"
                        ],
                    },
                },
            },
        }
        result["jsdiv"]["test"]["unlearning_method"].update(unlearning_method_resources)
    #############################################################################

    # ----

    ################################### MIA #####################################
    if "membership_inference_attack" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating MIA", end=" ")
        # Test Set
        test_mia_dict = get_membership_attack_prob(
            fabric=fabric,
            num_gpus=num_gpus,
            model=unlearned_model,
            retain_dataloader=retain_train_dataloader,
            forget_dataloader=forget_train_dataloader,
            test_dataloader=test_dataloader,
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating MIA")
        fabric.print(
            "The MIA value is: ",
            test_mia_dict["metric_value_dict"]["final_value"],
        )

        result["membership_inference_attack"] = {
            "test": {
                "unlearning_method": {
                    "mia": test_mia_dict["metric_value_dict"],
                },
                "eval_metric": {
                    "core_time_elapsed": test_mia_dict["core_time_dict"],
                    "memory_usage": test_mia_dict["memory_usage_dict"],
                    "power_consumption": test_mia_dict["power_consumption_dict"],
                },
            },
        }
        result["membership_inference_attack"]["test"]["unlearning_method"].update(
            unlearning_method_resources
        )
    #############################################################################

    # ----

    ############################### Activation Distance ############################
    if "activation_distance" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating Activation Distance", end=" ")

        # Whole Test Set
        fabric.print("\nCalculating Activation Distance on Whole Set", end=" ")
        act_distance_whole_dict = actv_dist(
            fabric=fabric,
            model1=reference_model,
            model2=unlearned_model,
            test_dataloader=test_dataloader,
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating Activation Distance on Whole Set")
        fabric.print(
            "The Activation Distance (Whole Set) value is: ",
            act_distance_whole_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("================================================")
        fabric.print("\nCalculating Activation Distance on Retain Set", end=" ")
        # Retain Test Set
        act_distance_retain_dict = actv_dist(
            fabric=fabric,
            model1=reference_model,
            model2=unlearned_model,
            test_dataloader=retain_test_dataloader,
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating Activation Distance on Retain Set")
        fabric.print(
            "The Activation Distance (Retain Set) value is: ",
            act_distance_retain_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("================================================")
        fabric.print("\nCalculating Activation Distance on Forget Set", end=" ")
        # Forget Test Set
        act_distance_forget_dict = actv_dist(
            fabric=fabric,
            model1=reference_model,
            model2=unlearned_model,
            test_dataloader=forget_test_dataloader,
            do_global_aggregation=False,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating Activation Distance on Forget Set")
        fabric.print(
            "The Activation Distance (Forget Set) value is: ",
            act_distance_forget_dict["metric_value_dict"]["final_value"],
        )

        result["activation_distance"] = {
            "test": {
                "unlearning_method": {
                    # Whole Test Set
                    "activation_distance_whole": act_distance_whole_dict[
                        "metric_value_dict"
                    ],
                    # Retain Test Set
                    "activation_distance_retain": act_distance_retain_dict[
                        "metric_value_dict"
                    ],
                    # Forget Test Set
                    "activation_distance_forget": act_distance_forget_dict[
                        "metric_value_dict"
                    ],
                },
                "eval_metric": {
                    "activation_distance_whole": {
                        "core_time_elapsed": act_distance_whole_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": act_distance_whole_dict[
                            "memory_usage_dict"
                        ],
                        "power_consumption": act_distance_whole_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "activation_distance_retain": {
                        "core_time_elapsed": act_distance_retain_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": act_distance_retain_dict[
                            "memory_usage_dict"
                        ],
                        "power_consumption": act_distance_retain_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "activation_distance_forget": {
                        "core_time_elapsed": act_distance_forget_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": act_distance_forget_dict[
                            "memory_usage_dict"
                        ],
                        "power_consumption": act_distance_forget_dict[
                            "power_consumption_dict"
                        ],
                    },
                },
            },
        }
        result["activation_distance"]["test"]["unlearning_method"].update(
            unlearning_method_resources
        )
    ################################################################################

    # ----

    ##### Layer-wise Distance || Weight Distance || Layer-wise Weight Difference ###
    if "layerwise_distance" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating Layer-wise Distance", end=" ")

        #########################################################
        # This is the only metric that is parameter-based whereas
        # all other metrics are data-based so they are automatically
        # handled by our DDP setting whereas this one is not.
        # So we need to manually distribute the parameters across ranks.
        # We do the calculation here as we do not want to add the
        # parameter distribution as part of the execution time of the metric
        #########################################################

        # FSDP requires gathering full parameters before iteration because they
        # are sharded across ranks. Element-wise comparison of local shards would
        # produce wrong results. We use FSDP.summon_full_params() as a context
        # manager to temporarily gather full params on all ranks, and call
        # lay_dist() INSIDE the context so parameters remain unsharded during
        # the computation. Once the context exits, params revert to sharded state.
        fsdp_active = _is_fsdp_wrapped(unlearned_model) or _is_fsdp_wrapped(reference_model)

        def _compute_layerwise_distance():
            # Get all parameter pairs
            param_pairs = list(
                zip(unlearned_model.named_parameters(), reference_model.named_parameters())
            )
            total_params = len(param_pairs)

            # Distribute parameters across ranks
            rank = fabric.global_rank
            world_size = fabric.world_size

            # Calculate which parameters this rank should process
            params_per_rank = total_params // world_size
            start_idx = rank * params_per_rank
            end_idx = start_idx + params_per_rank if rank < world_size - 1 else total_params

            return lay_dist(
                fabric=fabric,
                start_idx=start_idx,
                end_idx=end_idx,
                param_pairs=param_pairs,
                do_global_aggregation=do_global_aggregation,
                track_evaluation_resources=track_evaluation_resources,
            )

        if fsdp_active:
            # summon_full_params gathers full (non-sharded) params on every rank for
            # correct element-wise parameter comparison. Memory spikes briefly.
            with FSDP.summon_full_params(unlearned_model, writeback=False), \
                 FSDP.summon_full_params(reference_model, writeback=False):
                layerwise_distance_dict = _compute_layerwise_distance()
        else:
            layerwise_distance_dict = _compute_layerwise_distance()
        fabric.print("Finished Calculating Layer-wise Distance")
        fabric.print(
            "The Layer-wise Distance value is: ",
            layerwise_distance_dict["metric_value_dict"]["final_value"],
        )

        result["layerwise_distance"] = {
            "unlearning_method": {
                "layerwise_distance": layerwise_distance_dict["metric_value_dict"],
            },
            "eval_metric": {
                "core_time_elapsed": layerwise_distance_dict["core_time_dict"],
                "memory_usage": layerwise_distance_dict["memory_usage_dict"],
                "power_consumption": layerwise_distance_dict["power_consumption_dict"],
            },
        }
        result["layerwise_distance"]["unlearning_method"].update(
            unlearning_method_resources
        )
    ################################################################################

    ############################## Completeness #################################
    if "completeness" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating Completeness", end=" ")

        # Test Set
        fabric.print("\nCalculating Completeness on Whole Set", end=" ")
        test_completeness_dict = calculate_completeness(
            fabric=fabric,
            model1=unlearned_model,
            model2=reference_model,
            test_dataloader=test_dataloader,
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating Completeness on Whole Set")
        fabric.print(
            "The Completeness (Whole Set) value is: ",
            test_completeness_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("================================================")
        fabric.print("\nCalculating Completeness on Retain Set", end=" ")
        # Test Retain Set
        test_completeness_retain_dict = calculate_completeness(
            fabric=fabric,
            model1=unlearned_model,
            model2=reference_model,
            test_dataloader=retain_test_dataloader,
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating Completeness on Retain Set")
        fabric.print(
            "The Completeness (Retain Set) value is: ",
            test_completeness_retain_dict["metric_value_dict"]["final_value"],
        )

        fabric.print("================================================")
        fabric.print("\nCalculating Completeness on Forget Set", end=" ")
        # Test Forget Set
        test_completeness_forget_dict = calculate_completeness(
            fabric=fabric,
            model1=unlearned_model,
            model2=reference_model,
            test_dataloader=forget_test_dataloader,
            do_global_aggregation=False,
            track_evaluation_resources=track_evaluation_resources,
        )
        fabric.print("Finished Calculating Completeness on Forget Set")
        fabric.print(
            "The Completeness (Forget Set) value is: ",
            test_completeness_forget_dict["metric_value_dict"]["final_value"],
        )

        result["completeness"] = {
            "test": {
                "unlearning_method": {
                    # Whole Test Set
                    "completeness_whole": test_completeness_dict["metric_value_dict"],
                    # Retain Test Set
                    "completeness_retain": test_completeness_retain_dict[
                        "metric_value_dict"
                    ],
                    # Forget Test Set
                    "completeness_forget": test_completeness_forget_dict[
                        "metric_value_dict"
                    ],
                },
                "eval_metric": {
                    "completeness_whole": {
                        "core_time_elapsed": test_completeness_dict["core_time_dict"],
                        "memory_usage": test_completeness_dict["memory_usage_dict"],
                        "power_consumption": test_completeness_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "completeness_retain": {
                        "core_time_elapsed": test_completeness_retain_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": test_completeness_retain_dict[
                            "memory_usage_dict"
                        ],
                        "power_consumption": test_completeness_retain_dict[
                            "power_consumption_dict"
                        ],
                    },
                    "completeness_forget": {
                        "core_time_elapsed": test_completeness_forget_dict[
                            "core_time_dict"
                        ],
                        "memory_usage": test_completeness_forget_dict[
                            "memory_usage_dict"
                        ],
                        "power_consumption": test_completeness_forget_dict[
                            "power_consumption_dict"
                        ],
                    },
                },
            },
        }
        result["completeness"]["test"]["unlearning_method"].update(
            unlearning_method_resources
        )
    #############################################################################

    # ----

    ################################ Time ##################################
    if "time" in eval_metrics:
        fabric.print("================================================")
        fabric.print("\nCalculating Time", end=" ")
        # same applies about the retrained model's elapsed time
        # if it is None, then it means it's the retrained model's run so its elapsed time has not been saved yet
        assert retrain_time_elapsed_dict is not None, "Value must not be None"

        unlearn_time = core_time_dict["final_value"]

        time_dict = calculate_time(
            fabric, retrain_time_elapsed_dict["final_value"], unlearn_time
        )  # speedup

        fabric.print("Finished Calculating Time")

        result["time"] = {
            "unlearning_method": {
                "core_time_elapsed": unlearn_time,
                "speedup": time_dict["metric_value_dict"],
                # Note: The GPU IDs used for unlearning may differ from those used during inference.
                # When loading and evaluating saved unlearned model checkpoints, the recorded time
                # reflects the original unlearning run (with its specific GPU IDs), not the current inference GPUs.
                # This distinction is important: the reported time corresponds to the GPUs used during unlearning,
                # not those used for subsequent evaluation or inference.
                "gpu_ids": kwargs.get("model_unlearned_with_initial_gpu_ids", None),
            },
            "eval_metric": {
                "core_time_elapsed": time_dict["core_time_dict"],
                "memory_usage": time_dict["memory_usage_dict"],
                "power_consumption": time_dict["power_consumption_dict"],
            },
        }
        result["time"]["unlearning_method"].update(unlearning_method_resources)
    #############################################################################

    # ----

    ####################### Externally registered metrics #######################
    # Built-in metrics are handled by the branches above (byte-for-byte
    # unchanged). Any requested metric NOT recognised by those branches is
    # resolved through the registry and invoked here, so external metrics work
    # with no edits to this file. External metrics are expected to be decorated
    # with @track_evaluation_metric and thus return the standard envelope
    # {"metric_value_dict", "core_time_dict", "memory_usage_dict",
    #  "power_consumption_dict"}.
    from supreme.registry import external_metric_names, resolve_metric_location
    from supreme.utils.generic_utils import dynamic_method_call

    for metric_name in external_metric_names(eval_metrics):
        fabric.print("================================================")
        fabric.print(f"\nCalculating external metric: {metric_name}")
        module_name, attr_name = resolve_metric_location(metric_name)
        metric_output = dynamic_method_call(
            module_name=module_name,
            file_name=attr_name,
            fabric=fabric,
            num_gpus=num_gpus,
            original_model=original_model,
            unlearned_model=unlearned_model,
            unlearning_teacher=unlearning_teacher,
            reference_model=reference_model,
            retain_train_dataloader=retain_train_dataloader,
            retain_test_dataloader=retain_test_dataloader,
            forget_train_dataloader=forget_train_dataloader,
            forget_test_dataloader=forget_test_dataloader,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            do_global_aggregation=do_global_aggregation,
            track_evaluation_resources=track_evaluation_resources,
        )
        if metric_output is None:
            fabric.print(f"External metric '{metric_name}' returned no result.")
            continue
        result[metric_name] = {
            "unlearning_method": {
                metric_name: metric_output.get("metric_value_dict"),
            },
            "eval_metric": {
                "core_time_elapsed": metric_output.get("core_time_dict"),
                "memory_usage": metric_output.get("memory_usage_dict"),
                "power_consumption": metric_output.get("power_consumption_dict"),
            },
        }
        result[metric_name]["unlearning_method"].update(unlearning_method_resources)
    #############################################################################

    return result
