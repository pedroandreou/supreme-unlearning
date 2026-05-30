import os
import wandb
from typing import Optional
from dotenv import find_dotenv, load_dotenv
from wandb.integration.lightning.fabric import WandbLogger


def find_wandb_run_id(project_name: str, run_name: str, entity: Optional[str] = None) -> Optional[str]:
    """Find the WandB run ID for a given project and run name.

    Args:
        project_name: WandB project name
        run_name: WandB run display name (e.g., "retrain_seed60")
        entity: WandB entity/username. If None, uses default from wandb.api

    Returns:
        The run ID string if found, None otherwise
    """
    try:
        api = wandb.Api()
        if entity is None:
            entity = api.default_entity
        runs = api.runs(
            f"{entity}/{project_name}",
            filters={"display_name": run_name},
        )
        for run in runs:
            return run.id
        return None
    except Exception as e:
        print(f"Error looking up WandB run ID: {e}")
        return None


def initialize_wandb(fabric, config):
    """Initialize and configure Weights & Biases

    The config dict can optionally include:
        - actual_gpu_count: Override auto-detected GPU count in WandB metadata
        - actual_gpu_ids: List of GPU indices actually being used (e.g., [0, 1])
        - actual_gpu_type: Override auto-detected GPU type string
        - resume_if_exists: If True, look up existing run by name and resume it
          to append new metrics instead of creating a new run

    These are useful when running under SLURM or other schedulers where the
    visible GPUs may differ from what's actually allocated/used.
    """
    load_dotenv(find_dotenv())
    WANDB_KEY = os.getenv("WANDB_KEY")

    # # See https://docs.wandb.ai/support/run_wandb_offline/
    # os.environ["WANDB_MODE"] = "offline"

    wandb.login(key=WANDB_KEY)
    wandb.require("service")  # type: ignore

    # Initialize WandB logger
    # See documentation at
    # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb
    logger_kwargs = {
        "project": config["project_name"],
        "name": config["run_name"],
        "group": config.get("group_name"),
        "entity": os.getenv("WANDB_ENTITY") or None,
    }

    # Resume existing run if requested (to append new metrics to an existing run)
    if config.get("resume_if_exists"):
        existing_run_id = find_wandb_run_id(
            config["project_name"], config["run_name"]
        )
        # Fallback: try old naming format (e.g. `{method}_seed{U}`) if the
        # new format (`{method}_tseed{T}_useed{U}`) wasn't found and TRAINING_SEED is set.
        if not existing_run_id and os.environ.get("TRAINING_SEED"):
            for alt_name in config.get("alt_run_names", []):
                existing_run_id = find_wandb_run_id(
                    config["project_name"], alt_name
                )
                if existing_run_id:
                    fabric.print(
                        f"Found run under alternative name '{alt_name}'"
                    )
                    break
        if existing_run_id:
            logger_kwargs["id"] = existing_run_id
            logger_kwargs["resume"] = "must"
            fabric.print(
                f"Resuming existing WandB run: {existing_run_id} "
                f"(project: {config['project_name']}, name: {config['run_name']})"
            )
        else:
            fabric.print(
                f"No existing WandB run found for '{config['run_name']}' "
                f"in project '{config['project_name']}'. Creating new run."
            )

    wandb_logger = WandbLogger(**logger_kwargs)

    # # Define metrics to track min/max values
    # wandb_logger.experiment.define_metric("val_accuracy", summary="max")
    # wandb_logger.experiment.define_metric("train_loss", summary="min")

    wandb_logger.experiment.config.update(
        config["experiment_config"], allow_val_change=True
    )
    fabric.loggers.append(wandb_logger)

    return fabric


def sync_wandb(fabric):
    """Sync WandB run and handle cleanup"""
    try:
        if wandb.run:
            # Remove '/files' suffix from wandb run dir path by doing [:6]
            # https://github.com/wandb/wandb/issues/5764
            run_dir = wandb.run.dir[:-6]
            run_id = wandb.run.id

            fabric.print(f"\nW&B Run ID: {run_id}")
            fabric.print(f"W&B Run directory: {run_dir}")

            print(f"Syncing W&B run from directory: {run_dir}")
            os.system(f"wandb sync {run_dir}")

            fabric.print("\nIf you notice any errors in the log output file:")
            fabric.print("1. Check if the run was synced by visiting W&B web interface")
            fabric.print(
                "2. If not synced, try manually syncing using one of these commands:"
            )
            fabric.print(f"   - wandb sync {run_dir}")
            fabric.print(f'   - wandb sync --include-globs="*{run_id}*"')
            fabric.print(
                "3. If still having issues, check your internet connection and W&B authentication"
            )

            wandb.finish()
            fabric.print("The data has been logged to W&B successfully.")
        else:
            fabric.print("W&B run not initialized. Skipping sync.")
    except Exception as e:
        print(f"Error during W&B sync: {e}")


def check_wandb_run_exists(
    project_name: str, run_name: str, eval_metrics: list, entity: Optional[str] = None,
    max_retries: int = 5, retry_delay: float = 5.0,
):
    """
    Check if a WandB run with evaluation metrics already exists for the given configuration.

    Args:
        project_name: WandB project name (e.g., "R7_UNLEARNING_ResNet18_Cifar20_fullclass_vehicle2_precision_32-true")
        run_name: WandB run name (e.g., "retrain_seed60")
        eval_metrics: List of evaluation metric names to check for (e.g., ["accuracy", "zrf", "mia"])
        entity: WandB entity/username. If None, uses default from wandb.api
        max_retries: Number of retries on transient API errors (default: 5)
        retry_delay: Base delay in seconds between retries; exponential backoff with jitter (default: 5.0)

    Returns:
        tuple: (status, missing_metrics) where:
            - status: "all_exist" if all metrics present, "partial" if some missing, "none" if no metrics found
            - missing_metrics: list of high-level metric names that are missing (empty if all exist)
    """
    import time
    import random

    api_key = os.getenv("WANDB_API_KEY") or os.getenv("WANDB_KEY")

    # Retry the API query on transient errors (CommError, ValueError from wandb internals).
    runs = None
    last_api_error = None
    for attempt in range(max_retries + 1):
        try:
            api = wandb.Api(api_key=api_key, timeout=30)
            if entity is None:
                entity = api.default_entity
            runs_query = api.runs(
                f"{entity}/{project_name}",
                filters={"display_name": run_name},
            )
            runs = list(runs_query)
            break
        except (wandb.errors.CommError, ValueError) as e:
            # Project not yet created on WandB → no runs to find. Retrying is
            # pointless (project auto-creates only on first actual wandb.init),
            # so short-circuit instead of burning the backoff budget.
            if isinstance(e, ValueError) and "Could not find project" in str(e):
                return ("none", eval_metrics)
            last_api_error = e
            if attempt < max_retries:
                backoff = retry_delay * (2 ** attempt) + random.uniform(0, 3)
                print(
                    f"WandB API error (attempt {attempt + 1}/{max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                entity = None  # Re-resolve default_entity on retry in case of auth refresh
                continue

    # All retries exhausted without a successful API call - fail open (proceed with evaluation)
    if runs is None:
        print(
            f"WandB API check failed after {max_retries + 1} attempts for project "
            f"'{project_name}'. Will proceed with evaluation. Last error: {last_api_error}"
        )
        return ("none", eval_metrics)

    try:
        # Check if any runs exist
        if not runs:
            return ("none", eval_metrics)

        # Map evaluation metric names to their WandB logged names
        # WandB flattens nested dictionaries with dot notation (e.g., "accuracy.test.unlearning_method.whole_acc")
        # Metrics with "test" key depend on dataset type, others don't (see metrics_main.py lines 68-71)
        metric_name_mapping = {
            "accuracy": [
                "accuracy.test.unlearning_method.whole_acc",
                "accuracy.test.unlearning_method.retain_acc",
                "accuracy.test.unlearning_method.forget_acc",
                "loss.test.unlearning_method.whole_loss",
                "loss.test.unlearning_method.retain_loss",
                "loss.test.unlearning_method.forget_loss",
            ],
            "zrf": [
                "zrf.test.unlearning_method.initial_zrf",
                "zrf.test.unlearning_method.final_zrf",
            ],
            "jsdiv": [
                "jsdiv.test.unlearning_method.jsdiv_whole",
                "jsdiv.test.unlearning_method.jsdiv_retain",
                "jsdiv.test.unlearning_method.jsdiv_forget",
            ],
            "membership_inference_attack": ["membership_inference_attack.test.unlearning_method.mia"],
            "activation_distance": [
                "activation_distance.test.unlearning_method.activation_distance_whole",
                "activation_distance.test.unlearning_method.activation_distance_retain",
                "activation_distance.test.unlearning_method.activation_distance_forget",
            ],
            "layerwise_distance": ["layerwise_distance.unlearning_method.layerwise_distance"],
            "completeness": [
                "completeness.test.unlearning_method.completeness_whole",
                "completeness.test.unlearning_method.completeness_retain",
                "completeness.test.unlearning_method.completeness_forget",
            ],
            "time": [
                "time.unlearning_method.core_time_elapsed",
                "time.unlearning_method.speedup",
            ],
        }

        # Collect all WandB metric names to check, grouped by high-level metric name
        metric_to_wandb_keys = {}
        for metric in eval_metrics:
            if metric in metric_name_mapping:
                metric_to_wandb_keys[metric] = metric_name_mapping[metric]
            else:
                # If metric not in mapping, use it as-is
                metric_to_wandb_keys[metric] = [metric]

        def check_nested_key(dictionary, key_path):
            """
            Check if a nested key exists in a dictionary using dot notation.
            E.g., 'accuracy.test.unlearning_method.whole_acc' checks
            dictionary['accuracy']['test']['unlearning_method']['whole_acc']
            """
            keys = key_path.split('.')
            current = dictionary
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return False
            return True

        for run in runs:
            # Get the run's summary (contains final logged metrics)
            summary = run.summary._json_dict

            # Check which high-level metrics have ALL their WandB keys present
            metrics_present = []
            metrics_missing = []
            for metric, wandb_keys in metric_to_wandb_keys.items():
                if all(check_nested_key(summary, key) for key in wandb_keys):
                    metrics_present.append(metric)
                else:
                    metrics_missing.append(metric)

            if metrics_present and not metrics_missing:
                # ALL requested metrics exist - safe to skip
                print(
                    f"Found existing WandB run '{run_name}' in project '{project_name}' with ALL evaluation metrics: {eval_metrics}"
                )
                return ("all_exist", [])
            elif metrics_present and metrics_missing:
                # SOME metrics exist but others are missing - must re-evaluate to append
                print(
                    f"Found existing WandB run '{run_name}' in project '{project_name}' with partial metrics. "
                    f"Present: {metrics_present}. Missing: {metrics_missing}. Will proceed with evaluation to append missing metrics."
                )
                return ("partial", metrics_missing)

        # Run exists but no evaluation metrics found
        print(
            f"Found WandB run '{run_name}' in project '{project_name}', but no evaluation metrics detected for {eval_metrics}. Will proceed with evaluation."
        )
        return ("none", eval_metrics)

    except wandb.errors.CommError as e:
        # Project doesn't exist or no access
        print(
            f"WandB project '{project_name}' not found or no access. Will proceed with evaluation. Error: {e}"
        )
        return ("none", eval_metrics)
    except Exception as e:
        # Other errors (network issues, authentication, etc.)
        print(
            f"Error checking WandB for existing run: {e}. Will proceed with evaluation."
        )
        return ("none", eval_metrics)


def main():
    """
    Command-line interface for checking if WandB evaluation results exist.
    Called from run_local.sh before launching the full evaluation to avoid unnecessary
    model/data loading when results are already logged.

    Exit codes:
        0: Results exist in WandB (should skip evaluation)
        1: Results don't exist (should proceed with evaluation)
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check if WandB evaluation results exist")

    # Minimal arguments needed to construct project_name and run_name
    parser.add_argument("-method", type=str, required=True)
    parser.add_argument("-net", type=str, required=True)
    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-type_of_unlearning_strategy", type=str, required=True)
    parser.add_argument("-seed", type=int, required=True)
    parser.add_argument("-precision", type=str, required=True)
    parser.add_argument("-eval_metrics", type=str, required=True)

    # Strategy-specific arguments
    parser.add_argument("-forget_class_name", type=str, default=None)
    parser.add_argument("-forget_subclass_name", type=str, default=None)
    parser.add_argument("-forget_perc", type=float, default=None)
    
    # Additional arguments that might be passed but aren't needed for checking
    parser.add_argument("-classes", type=int, default=None, help="Number of classes (not used for WandB check)")
    parser.add_argument("-superclasses", type=int, default=None, help="Number of superclasses (not used for WandB check)")
    parser.add_argument("-subclasses", type=int, default=None, help="Number of subclasses (not used for WandB check)")

    args = parser.parse_args()

    # Extract parameters
    method_name = args.method.lower()
    model_name = args.net
    dataset_name = args.dataset
    type_of_unlearning_strategy = args.type_of_unlearning_strategy
    seed = args.seed
    precision = args.precision
    eval_metrics = args.eval_metrics.split(",")

    # Determine forget_class_name based on strategy
    if type_of_unlearning_strategy == "random_":
        forget_class_name = f"{args.forget_perc}perc"
    elif type_of_unlearning_strategy == "fullclass":
        forget_class_name = args.forget_class_name
    elif type_of_unlearning_strategy == "subclass":
        forget_class_name = args.forget_subclass_name
    else:
        print(f"Unknown unlearning strategy: {type_of_unlearning_strategy}", file=sys.stderr)
        sys.exit(1)

    # Get experiment scenario from environment (same as unlearn_main.py)
    experiment_scenario = os.getenv("SCALABLE_EXPERIMENT_SCENARIO", "")

    # Construct project_name (same logic as unlearn_main.py lines 1314-1334)
    # Use WANDB_PROJECT_PREFIX environment variable (default: R7) for flexibility
    wandb_project_prefix = os.getenv("WANDB_PROJECT_PREFIX", "R14")
    project_name_parts = list(
        filter(
            None,
            [
                f"{wandb_project_prefix}_UNLEARNING",
                experiment_scenario,
                model_name,
                dataset_name,
                type_of_unlearning_strategy,
                forget_class_name,
                f"precision_{precision}",
            ],
        )
    )
    project_name = "_".join(project_name_parts)

    # Construct run_name. Mirror the three-step convention in unlearn_main.py:
    #   1. `{method}_seed{U}`                       (no TRAINING_SEED set)
    #   2. `{method}_tseed{T}_useed{U}`             (J>1, K=1)
    #   3. `{method}_tseed{T}_useed{U}_eseed{E}`    (K>1 eval, -seed = s_e ≠ s_u)
    _training_seed_env = os.environ.get('TRAINING_SEED')
    _unlearning_seed_env = os.environ.get('UNLEARNING_SEED')
    if _training_seed_env and _unlearning_seed_env and int(_unlearning_seed_env) != int(seed):
        run_name = (
            f"{method_name}_tseed{_training_seed_env}"
            f"_useed{_unlearning_seed_env}_eseed{seed}"
        )
    elif _training_seed_env:
        run_name = f"{method_name}_tseed{_training_seed_env}_useed{seed}"
    else:
        run_name = f"{method_name}_seed{seed}"

    # Check if results exist in WandB
    status, missing_metrics = check_wandb_run_exists(project_name, run_name, eval_metrics)

    # Exit codes:
    #   0: All metrics exist - skip evaluation
    #   1: No metrics exist - proceed with all requested metrics
    #   2: Partial metrics - proceed with only missing metrics (printed to stdout as MISSING_METRICS=...)
    if status == "all_exist":
        sys.exit(0)
    elif status == "partial":
        # Print missing metrics so run_local.sh can capture and use only those
        print(f"MISSING_METRICS={','.join(missing_metrics)}")
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
