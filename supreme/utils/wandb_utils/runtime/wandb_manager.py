#!/usr/bin/env python3
"""
Unified WandB Management Tool for Machine Unlearning Experiments.

This script provides commands for managing WandB runs, ordered from
informational/discovery commands to action/deletion commands:

  INFORMATIONAL:
    list-projects    : List all expected project names
    find-duplicates  : Find duplicate runs in projects
    find-missing     : Find missing runs for specified seeds
    generate-report  : Generate detailed duplicate report (JSON)

  ACTIONS:
    cleanup-empty    : Delete runs with no evaluation metrics
    delete-duplicates: Delete identical duplicate runs

Usage:
    python wandb_manager.py <command> [options]

Examples:
    # List all expected project names
    python wandb_manager.py list-projects
    python wandb_manager.py list-projects --strategy fullclass

    # Find duplicates across all projects
    python wandb_manager.py find-duplicates --all

    # Find duplicates in a specific project
    python wandb_manager.py find-duplicates --project "..."

    # Find missing runs for seeds 60-69 (excluding 66)
    python wandb_manager.py find-missing --training-seeds 60,61,62,63,64,65,67,68,69
    python wandb_manager.py find-missing --training-seeds 60-69 --exclude 66

    # Generate duplicate report (JSON)
    python wandb_manager.py generate-report --output report.json

    # Find empty runs in a project (dry run)
    python wandb_manager.py cleanup-empty --project "R7_UNLEARNING_ViT_Cifar100_fullclass_lamp_precision_32-true"

    # Delete empty runs
    python wandb_manager.py cleanup-empty --project "..." --delete

    # Delete identical duplicates
    python wandb_manager.py delete-duplicates --project "..." --confirm
"""

import argparse
import sys
import json
import os
from collections import defaultdict
from typing import List, Tuple
from datetime import datetime

try:
    import wandb
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
except ImportError:
    print(
        "Error: Required packages not installed. Run: pip install wandb python-dotenv"
    )
    sys.exit(1)


# ==============================================================================
# Configuration
# ==============================================================================

# Project name prefix (can be R6, R7, etc.)
PROJECT_PREFIX = os.getenv("WANDB_PROJECT_PREFIX", "R6")

# Default evaluation metrics to check
DEFAULT_EVAL_METRICS = [
    "accuracy",
    "zrf",
    "activation_distance",
    "completeness",
    "jsdiv",
    "layerwise_distance",
    "time",
    "membership_inference_attack",
]

# Metric name mappings (how metrics are logged in WandB)
METRIC_NAME_MAPPING = {
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
    "jsdiv": ["jsdiv.test.unlearning_method.jsdiv"],
    "membership_inference_attack": [
        "membership_inference_attack.test.unlearning_method.mia"
    ],
    "activation_distance": [
        "activation_distance.unlearning_method.activation_distance"
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

# Dataset configurations for project name generation
DATASETS_FULLCLASS = {
    "Cifar20": ["vehicle2", "veg", "people", "electrical_devices", "natural_scenes"],
    "Cifar100": ["rocket", "mushroom", "baby", "lamp", "sea"],
    "PinsFaceRecognition": ["1", "10", "20", "30", "40"],
    "Caltech101": ["airplanes", "car_side", "chair", "elephant", "lamp"],
}

DATASETS_SUBCLASS = {
    "Cifar20": ["rocket", "mushroom", "baby", "lamp", "sea"],
}

DATASETS_RANDOM = {
    "Cifar10": ["0.001", "0.005", "0.01", "0.05", "0.1"],
    "PinsFaceRecognition": ["0.001", "0.005", "0.01", "0.05", "0.1"],
    "Caltech101": ["0.001", "0.005", "0.01", "0.05", "0.1"],
}

MODELS = ["ResNet18", "ViT"]
PRECISION = "32-true"

# Methods (canonical names as they appear in WandB run names)
# From project_config.py: baselines + unlearning_methods

# Core methods (the main methods typically used in experiments)
METHODS_CORE = [
    "original",
    "retrain",  # baselines
    "finetune",
    "bad_teacher",
    "random_labeling",
    "unsir",
    "ssd",
    "lfssd",  # core unlearning methods
]
METHODS_CORE_RANDOM = [
    "original",
    "retrain",  # baselines
    "finetune",
    "bad_teacher",
    "random_labeling",
    "ssd",
    "lfssd",  # core unlearning methods (no UNSIR for random)
]

# All methods including additional ones
METHODS_STANDARD = METHODS_CORE + [
    "neg_grad",  # additional methods
]
METHODS_RANDOM = METHODS_CORE_RANDOM + [
    "neg_grad",  # additional methods
]


def get_methods_for_strategy(strategy: str, methods_mode: str = "all") -> List[str]:
    """Get methods list based on strategy and methods mode.

    Args:
        strategy: The unlearning strategy (fullclass, subclass, random, random_)
        methods_mode: "all" for all methods, "core" for just the 8 core methods
    """
    if strategy in ["random", "random_"]:
        if methods_mode == "core":
            return METHODS_CORE_RANDOM
        return METHODS_RANDOM
    else:
        if methods_mode == "core":
            return METHODS_CORE
        return METHODS_STANDARD


# ==============================================================================
# Utility Functions
# ==============================================================================


def check_nested_key(dictionary: dict, key_path: str) -> bool:
    """Check if a nested key exists in a dictionary using dot notation."""
    keys = key_path.split(".")
    current = dictionary
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False
    return True


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary using dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_evaluation_metrics(summary: dict) -> dict:
    """Extract evaluation metrics from summary, excluding resource tracking."""
    flat_summary = flatten_dict(summary)

    key_patterns = [
        "accuracy.test.unlearning_method.whole_acc",
        "accuracy.test.unlearning_method.retain_acc",
        "accuracy.test.unlearning_method.forget_acc",
        "zrf.test.unlearning_method.initial_zrf",
        "zrf.test.unlearning_method.final_zrf",
        "jsdiv.test.unlearning_method.jsdiv",
        "membership_inference_attack.test.unlearning_method.mia",
        "activation_distance.unlearning_method.activation_distance",
        "layerwise_distance.unlearning_method.layerwise_distance",
        "completeness.test.unlearning_method.completeness_whole",
        "completeness.test.unlearning_method.completeness_retain",
        "completeness.test.unlearning_method.completeness_forget",
        "time.unlearning_method.speedup",
    ]

    metrics = {}
    for pattern in key_patterns:
        for key, value in flat_summary.items():
            if pattern in key and ("final_value" in key or key == pattern):
                base_key = pattern
                if base_key not in metrics or "final_value" in key:
                    # Extract actual value if nested in {'final_value': X}
                    if isinstance(value, dict) and "final_value" in value:
                        metrics[base_key] = value["final_value"]
                    else:
                        metrics[base_key] = value

    return metrics


def compare_metrics(
    metrics1: dict, metrics2: dict, tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
    """Compare two metric dictionaries.

    Args:
        metrics1: First metric dictionary
        metrics2: Second metric dictionary
        tolerance: Tolerance for float comparison (default 1e-6)

    Returns:
        Tuple of (are_identical, list_of_differences)
    """
    # Keys to skip when comparing (timing varies between runs)
    SKIP_KEYS = {
        "time.unlearning_method.core_time_elapsed",
        "time.unlearning_method.speedup",
    }

    all_keys = set(metrics1.keys()) | set(metrics2.keys())
    differences = []

    for key in sorted(all_keys):
        # Skip timing metrics that always differ
        if key in SKIP_KEYS:
            continue

        val1 = metrics1.get(key)
        val2 = metrics2.get(key)

        if val1 is None and val2 is None:
            continue
        if val1 is None or val2 is None:
            differences.append(f"{key}: {val1} vs {val2} (one is missing)")
            continue

        if isinstance(val1, float) and isinstance(val2, float):
            if abs(val1 - val2) > tolerance:
                differences.append(f"{key}: {val1} vs {val2}")
        elif val1 != val2:
            differences.append(f"{key}: {val1} vs {val2}")

    return len(differences) == 0, differences


def has_evaluation_metrics(run, eval_metrics: List[str] = None) -> bool:
    """Check if a run has any evaluation metrics."""
    if eval_metrics is None:
        eval_metrics = DEFAULT_EVAL_METRICS

    try:
        summary = run.summary._json_dict

        if not summary or len(summary) == 0:
            return False

        # Collect all WandB metric names to check
        wandb_metrics_to_check = []
        for metric in eval_metrics:
            if metric in METRIC_NAME_MAPPING:
                wandb_metrics_to_check.extend(METRIC_NAME_MAPPING[metric])
            else:
                wandb_metrics_to_check.append(metric)

        # Check if at least one metric exists
        for metric in wandb_metrics_to_check:
            if check_nested_key(summary, metric):
                return True

        # Fallback: check for metric-like keys
        summary_keys = list(summary.keys())
        metric_indicators = [
            "acc",
            "loss",
            "zrf",
            "mia",
            "distance",
            "completeness",
            "time",
            "jsdiv",
        ]
        if any(
            key
            for key in summary_keys
            if any(ind in key.lower() for ind in metric_indicators)
        ):
            return True

        return False
    except Exception:
        return False


def generate_all_project_names(prefix: str = None) -> List[str]:
    """Generate all possible project names based on configuration."""
    if prefix is None:
        prefix = PROJECT_PREFIX

    projects = []

    # Fullclass projects
    for dataset, classes in DATASETS_FULLCLASS.items():
        for model in MODELS:
            for forget_class in classes:
                projects.append(
                    f"{prefix}_UNLEARNING_{model}_{dataset}_fullclass_{forget_class}_precision_{PRECISION}"
                )

    # Subclass projects
    for dataset, classes in DATASETS_SUBCLASS.items():
        for model in MODELS:
            for forget_class in classes:
                projects.append(
                    f"{prefix}_UNLEARNING_{model}_{dataset}_subclass_{forget_class}_precision_{PRECISION}"
                )

    # Random projects
    for dataset, percs in DATASETS_RANDOM.items():
        for model in MODELS:
            for perc in percs:
                projects.append(
                    f"{prefix}_UNLEARNING_{model}_{dataset}_random_{perc}perc_precision_{PRECISION}"
                )

    return projects


def get_api_and_entity(entity: str = None):
    """Initialize WandB API and get entity."""
    api = wandb.Api()
    if entity is None:
        entity = api.default_entity
    return api, entity


# ==============================================================================
# Command: cleanup-empty
# ==============================================================================


def cmd_cleanup_empty(args):
    """Find and delete empty WandB runs."""
    api, entity = get_api_and_entity(args.entity)

    if not args.project:
        print("Error: --project is required")
        return 1

    dry_run = not args.delete

    print("=" * 70)
    print("CLEANUP EMPTY RUNS" + (" (DRY RUN)" if dry_run else " (DELETE MODE)"))
    print("=" * 70)

    try:
        project_path = f"{entity}/{args.project}"
        print(f"Project: {project_path}")

        filters = {}
        if args.run_name:
            filters["display_name"] = args.run_name
            print(f"Filter: run_name = {args.run_name}")

        runs = api.runs(project_path, filters=filters)
        print(f"Found {len(runs)} total run(s)\n")

        empty_runs = []
        for run in runs:
            if not has_evaluation_metrics(run):
                empty_runs.append(run)
                print(f"  Empty: {run.id} ({run.name}) - State: {run.state}")
            else:
                print(f"  Has metrics: {run.id} ({run.name})")

        if empty_runs:
            print(f"\nFound {len(empty_runs)} empty run(s)")

            if dry_run:
                print("\nDRY RUN - No runs deleted. Use --delete to remove them.")
            else:
                print(f"\nDeleting {len(empty_runs)} empty run(s)...")
                deleted = 0
                for run in empty_runs:
                    try:
                        run.delete()
                        print(f"  Deleted: {run.id}")
                        deleted += 1
                    except Exception as e:
                        print(f"  Failed: {run.id} - {e}")
                print(f"\nDeleted {deleted}/{len(empty_runs)} runs")
        else:
            print("\nNo empty runs found.")

        return 0
    except wandb.errors.CommError as e:
        print(f"Error accessing project: {e}")
        return 1


# ==============================================================================
# Command: find-duplicates
# ==============================================================================


def cmd_find_duplicates(args):
    """Find duplicate runs."""
    api, entity = get_api_and_entity(args.entity)

    if args.all:
        project_names = generate_all_project_names(args.prefix)
        print(f"Scanning {len(project_names)} projects for duplicates...\n")
    elif args.project:
        project_names = [args.project]
    else:
        print("Error: Specify --project or --all")
        return 1

    print("=" * 70)
    print("FIND DUPLICATE RUNS")
    print("=" * 70)

    total_duplicates = 0
    projects_with_duplicates = []

    for project_name in project_names:
        try:
            project_path = f"{entity}/{project_name}"
            runs = api.runs(project_path)

            runs_by_name = defaultdict(list)
            for run in runs:
                runs_by_name[run.name].append(run)

            duplicates = {
                name: runs_list
                for name, runs_list in runs_by_name.items()
                if len(runs_list) > 1
            }

            if duplicates:
                projects_with_duplicates.append((project_name, duplicates))
                total_duplicates += len(duplicates)

                if not args.all or args.verbose:
                    print(f"\n{project_name}: {len(duplicates)} duplicate group(s)")
                    for run_name, runs_list in duplicates.items():
                        print(f"  {run_name}: {len(runs_list)} duplicate(s)")
                        for run in runs_list:
                            print(
                                f"    - {run.id} (created: {run.created_at}, state: {run.state})"
                            )
        except (wandb.errors.CommError, ValueError):
            pass  # Project doesn't exist

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Projects checked: {len(project_names)}")
    print(f"Projects with duplicates: {len(projects_with_duplicates)}")
    print(f"Total duplicate groups: {total_duplicates}")

    return 0


# ==============================================================================
# Command: delete-duplicates
# ==============================================================================


def cmd_delete_duplicates(args):
    """Delete identical duplicate runs (keep oldest)."""
    api, entity = get_api_and_entity(args.entity)

    if args.all:
        project_names = generate_all_project_names(args.prefix)
    elif args.project:
        project_names = [args.project]
    else:
        print("Error: Specify --project or --all")
        return 1

    if not args.confirm:
        print("=" * 70)
        print("DRY RUN - Use --confirm to actually delete duplicates")
        print("=" * 70)
    else:
        print("=" * 70)
        print("DELETE MODE - Identical duplicates will be permanently deleted!")
        print("=" * 70)
        if not args.yes:
            response = input("Are you sure? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return 0

    total_deleted = 0
    total_kept = 0

    for project_name in project_names:
        try:
            project_path = f"{entity}/{project_name}"
            runs = api.runs(project_path)

            runs_by_name = defaultdict(list)
            for run in runs:
                runs_by_name[run.name].append(run)

            duplicates = {
                name: runs_list
                for name, runs_list in runs_by_name.items()
                if len(runs_list) > 1
            }

            if duplicates:
                print(f"\n{project_name}:")

                for run_name, runs_list in duplicates.items():
                    # Extract metrics and check if identical
                    run_metrics = []
                    for run in runs_list:
                        try:
                            summary = run.summary._json_dict
                            metrics = extract_evaluation_metrics(summary)
                            run_metrics.append((run, metrics))
                        except Exception:
                            run_metrics.append((run, {}))

                    # Check if all identical
                    all_identical = True
                    for i in range(len(run_metrics)):
                        for j in range(i + 1, len(run_metrics)):
                            _, m1 = run_metrics[i]
                            _, m2 = run_metrics[j]
                            if not compare_metrics(m1, m2)[0]:
                                all_identical = False
                                break
                        if not all_identical:
                            break

                    if all_identical:
                        runs_sorted = sorted(runs_list, key=lambda r: r.created_at)
                        oldest = runs_sorted[0]
                        to_delete = runs_sorted[1:]

                        print(
                            f"  {run_name}: {len(runs_list)} duplicates (all identical)"
                        )
                        print(f"    Keep: {oldest.id}")
                        total_kept += 1

                        for run in to_delete:
                            if args.confirm:
                                try:
                                    run.delete()
                                    print(f"    Deleted: {run.id}")
                                    total_deleted += 1
                                except Exception as e:
                                    print(f"    Failed: {run.id} - {e}")
                            else:
                                print(f"    Would delete: {run.id}")
                                total_deleted += 1
                    else:
                        print(
                            f"  {run_name}: {len(runs_list)} duplicates (DIFFERENT - skipped)"
                        )
        except (wandb.errors.CommError, ValueError):
            pass  # Project doesn't exist

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Runs kept: {total_kept}")
    print(f"Runs {'deleted' if args.confirm else 'would delete'}: {total_deleted}")

    return 0


# ==============================================================================
# Command: generate-report
# ==============================================================================


def cmd_generate_report(args):
    """Generate detailed duplicate report."""
    api, entity = get_api_and_entity(args.entity)

    project_names = generate_all_project_names(args.prefix)

    print("=" * 70)
    print("GENERATING DUPLICATE REPORT")
    print("=" * 70)
    print(f"Scanning {len(project_names)} projects...\n")

    identical_duplicates = []
    different_duplicates = []

    for project_name in project_names:
        try:
            project_path = f"{entity}/{project_name}"
            runs = api.runs(project_path)

            runs_by_name = defaultdict(list)
            for run in runs:
                runs_by_name[run.name].append(run)

            duplicates = {
                name: runs_list
                for name, runs_list in runs_by_name.items()
                if len(runs_list) > 1
            }

            if duplicates:
                for run_name, runs_list in duplicates.items():
                    run_metrics = []
                    for run in runs_list:
                        try:
                            summary = run.summary._json_dict
                            metrics = extract_evaluation_metrics(summary)
                            run_metrics.append((run, metrics))
                        except Exception:
                            run_metrics.append((run, {}))

                    # Check if identical
                    all_identical = True
                    for i in range(len(run_metrics)):
                        for j in range(i + 1, len(run_metrics)):
                            _, m1 = run_metrics[i]
                            _, m2 = run_metrics[j]
                            if not compare_metrics(m1, m2)[0]:
                                all_identical = False
                                break
                        if not all_identical:
                            break

                    runs_sorted = sorted(runs_list, key=lambda r: r.created_at)
                    oldest = runs_sorted[0]
                    to_delete = runs_sorted[1:]

                    entry = {
                        "project": project_name,
                        "run_name": run_name,
                        "total_duplicates": len(runs_list),
                        "identical": all_identical,
                        "oldest_run_id": oldest.id,
                        "oldest_created": str(oldest.created_at),
                        "runs_to_delete": [
                            {
                                "run_id": run.id,
                                "created": str(run.created_at),
                                "state": run.state,
                            }
                            for run in to_delete
                        ],
                    }

                    if all_identical:
                        identical_duplicates.append(entry)
                    else:
                        different_duplicates.append(entry)
        except (wandb.errors.CommError, ValueError):
            pass  # Project doesn't exist

    # Build report
    report = {
        "generated_at": datetime.now().isoformat(),
        "entity": entity,
        "projects_scanned": len(project_names),
        "summary": {
            "identical_duplicate_groups": len(identical_duplicates),
            "different_duplicate_groups": len(different_duplicates),
            "runs_to_delete": sum(
                d["total_duplicates"] - 1 for d in identical_duplicates
            ),
        },
        "identical_duplicates": identical_duplicates,
        "different_duplicates": different_duplicates,
    }

    # Output
    output_file = (
        args.output
        or f"wandb_duplicate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_file}\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Identical duplicate groups: {len(identical_duplicates)}")
    print(f"Different duplicate groups: {len(different_duplicates)}")
    print(f"Runs that can be safely deleted: {report['summary']['runs_to_delete']}")

    return 0


# ==============================================================================
# Command: list-projects
# ==============================================================================


def cmd_list_projects(args):
    """List all expected project names."""
    print("=" * 70)
    print("EXPECTED PROJECT NAMES")
    print("=" * 70)

    prefix = args.prefix or PROJECT_PREFIX

    if args.strategy == "all" or args.strategy == "fullclass":
        print(f"\nFULLCLASS PROJECTS ({prefix}):")
        for dataset, classes in DATASETS_FULLCLASS.items():
            for model in MODELS:
                for forget_class in classes:
                    print(
                        f"  {prefix}_UNLEARNING_{model}_{dataset}_fullclass_{forget_class}_precision_{PRECISION}"
                    )

    if args.strategy == "all" or args.strategy == "subclass":
        print(f"\nSUBCLASS PROJECTS ({prefix}):")
        for dataset, classes in DATASETS_SUBCLASS.items():
            for model in MODELS:
                for forget_class in classes:
                    print(
                        f"  {prefix}_UNLEARNING_{model}_{dataset}_subclass_{forget_class}_precision_{PRECISION}"
                    )

    if args.strategy == "all" or args.strategy == "random":
        print(f"\nRANDOM PROJECTS ({prefix}):")
        for dataset, percs in DATASETS_RANDOM.items():
            for model in MODELS:
                for perc in percs:
                    print(
                        f"  {prefix}_UNLEARNING_{model}_{dataset}_random_{perc}perc_precision_{PRECISION}"
                    )

    projects = generate_all_project_names(prefix)
    print(f"\nTotal: {len(projects)} projects")

    return 0


# ==============================================================================
# Command: find-missing
# ==============================================================================


def parse_seeds(seeds_str: str, exclude_str: str = None) -> List[int]:
    """
    Parse seeds from string format.

    Supports:
    - Comma-separated: "60,61,62,63"
    - Range: "60-69"
    - Combined: "60-65,67-69"
    """
    seeds = set()

    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            # Range format
            start, end = part.split("-")
            seeds.update(range(int(start), int(end) + 1))
        else:
            seeds.add(int(part))

    # Handle exclusions
    if exclude_str:
        for part in exclude_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                for s in range(int(start), int(end) + 1):
                    seeds.discard(s)
            else:
                seeds.discard(int(part))

    return sorted(seeds)


def get_project_name_for_missing(
    prefix: str, model: str, dataset: str, strategy: str, class_name: str
) -> str:
    """Generate WandB project name for find-missing command."""
    if strategy in ["random", "random_"]:
        # Note: random_ uses double underscore in project names: random__
        return f"{prefix}_UNLEARNING_{model}_{dataset}_random__{class_name}perc_precision_{PRECISION}"
    else:
        return f"{prefix}_UNLEARNING_{model}_{dataset}_{strategy}_{class_name}_precision_{PRECISION}"


def cmd_find_missing(args):
    """Find missing WandB runs for specified seeds."""
    api, entity = get_api_and_entity(args.entity)
    prefix = args.prefix or PROJECT_PREFIX

    # Parse seeds
    seeds = parse_seeds(args.training_seeds, args.exclude)
    if not seeds:
        print("Error: No seeds specified after exclusions")
        return 1

    print("=" * 80)
    print("FINDING MISSING WANDB RUNS")
    print(f"Entity: {entity}")
    print(f"Project prefix: {prefix}")
    print(f"Seeds: {seeds}")
    print("=" * 80)
    print()

    # Build dataset configurations based on strategy filter
    datasets_config = {}

    if args.strategy in ["all", "fullclass"]:
        datasets_config["fullclass"] = {
            dataset: {"classes": classes, "models": MODELS}
            for dataset, classes in DATASETS_FULLCLASS.items()
        }

    if args.strategy in ["all", "subclass"]:
        datasets_config["subclass"] = {
            dataset: {"classes": classes, "models": MODELS}
            for dataset, classes in DATASETS_SUBCLASS.items()
        }

    if args.strategy in ["all", "random"]:
        datasets_config["random_"] = {
            dataset: {"classes": percs, "models": MODELS}
            for dataset, percs in DATASETS_RANDOM.items()
        }

    # Query WandB for existing runs
    print("Querying WandB for existing runs...")
    existing_runs = defaultdict(set)

    for strategy, datasets in datasets_config.items():
        for dataset, config in datasets.items():
            for model in config["models"]:
                for class_name in config["classes"]:
                    project_name = get_project_name_for_missing(
                        prefix, model, dataset, strategy, class_name
                    )
                    project_path = f"{entity}/{project_name}"

                    try:
                        runs = api.runs(project_path)
                        for run in runs:
                            existing_runs[project_name].add(run.name)
                        print(
                            f"  Found {len(existing_runs[project_name])} runs in {project_name}"
                        )
                    except (wandb.errors.CommError, ValueError):
                        print(f"  Project not found or empty: {project_name}")

    # Find missing runs
    print()
    print("=" * 80)
    print("ANALYZING MISSING RUNS")
    print("=" * 80)

    # Determine methods mode
    methods_mode = getattr(args, "methods", "core")  # default to core
    print(f"Methods mode: {methods_mode}")

    missing = []
    for strategy, datasets in datasets_config.items():
        methods = get_methods_for_strategy(strategy, methods_mode)

        for dataset, config in datasets.items():
            for model in config["models"]:
                for class_name in config["classes"]:
                    project_name = get_project_name_for_missing(
                        prefix, model, dataset, strategy, class_name
                    )
                    project_runs = existing_runs.get(project_name, set())

                    for method in methods:
                        for seed in seeds:
                            run_name = f"{method}_seed{seed}"
                            if run_name not in project_runs:
                                missing.append(
                                    (strategy, dataset, model, class_name, method, seed)
                                )

    if not missing:
        print(
            f"\nNo missing runs found! All experiments for seeds {seeds} are complete."
        )
        return 0

    # Summary
    print(f"\nTotal missing run combinations: {len(missing)}")

    # Group by strategy
    by_strategy = defaultdict(list)
    for item in missing:
        by_strategy[item[0]].append(item)

    print("\nMissing runs by strategy:")
    for strategy, items in sorted(by_strategy.items()):
        print(f"  {strategy}: {len(items)} missing")

    # Group by seed
    by_seed = defaultdict(list)
    for item in missing:
        by_seed[item[5]].append(item)

    print("\nMissing runs by seed:")
    for seed, items in sorted(by_seed.items()):
        print(f"  seed{seed}: {len(items)} missing")

    # Group by dataset
    by_dataset = defaultdict(list)
    for item in missing:
        by_dataset[item[1]].append(item)

    print("\nMissing runs by dataset:")
    for dataset, items in sorted(by_dataset.items()):
        print(f"  {dataset}: {len(items)} missing")

    # Generate submission commands
    print()
    print("=" * 80)
    print("RECOMMENDED SUBMISSION COMMANDS")
    print("=" * 80)
    print()

    # Group by (strategy, dataset) for efficient submission
    grouped = {}
    for strategy, dataset, model, class_name, method, seed in missing:
        key = (strategy, dataset)
        if key not in grouped:
            grouped[key] = {"models": set(), "seeds": set()}
        grouped[key]["models"].add(model)
        grouped[key]["seeds"].add(seed)

    commands = []
    for (strategy, dataset), info in sorted(grouped.items()):
        models = sorted(info["models"])
        missing_seeds = sorted(info["seeds"])

        models_str = ",".join(models)
        seeds_str = ",".join(map(str, missing_seeds))

        cmd = f"./supreme/run_slurm.sh --strategies {strategy} --datasets {dataset} --models {models_str} --training-seeds {seeds_str}"
        commands.append(cmd)

    for i, cmd in enumerate(commands, 1):
        print(f"# Command {i}")
        print(cmd)
        print()

    # Detailed missing list
    if args.verbose:
        print("=" * 80)
        print("DETAILED MISSING RUNS")
        print("=" * 80)
        for strategy, dataset, model, class_name, method, seed in missing:
            print(f"  {strategy}/{dataset}/{model}/{class_name}/{method}_seed{seed}")
    else:
        print("=" * 80)
        print(f"DETAILED MISSING RUNS (first 50 of {len(missing)})")
        print("=" * 80)
        for i, (strategy, dataset, model, class_name, method, seed) in enumerate(
            missing[:50]
        ):
            print(f"  {strategy}/{dataset}/{model}/{class_name}/{method}_seed{seed}")
        if len(missing) > 50:
            print(f"  ... and {len(missing) - 50} more (use --verbose to see all)")

    return 0


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified WandB Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -------------------------------------------------------------------------
    # INFORMATIONAL COMMANDS (discovery/read-only)
    # -------------------------------------------------------------------------

    # list-projects
    p_list = subparsers.add_parser("list-projects", help="List expected project names")
    p_list.add_argument("--prefix", default=PROJECT_PREFIX, help="Project prefix")
    p_list.add_argument(
        "--strategy", choices=["all", "fullclass", "subclass", "random"], default="all"
    )

    # find-duplicates
    p_find = subparsers.add_parser("find-duplicates", help="Find duplicate runs")
    p_find.add_argument("--project", help="Specific project")
    p_find.add_argument("--all", action="store_true", help="Scan all projects")
    p_find.add_argument("--entity", help="WandB entity")
    p_find.add_argument(
        "--prefix", default=PROJECT_PREFIX, help="Project prefix (R6, R7)"
    )
    p_find.add_argument("--verbose", "-v", action="store_true", help="Show details")

    # find-missing
    p_missing = subparsers.add_parser(
        "find-missing", help="Find missing runs for specified seeds"
    )
    p_missing.add_argument(
        "--training-seeds",
        dest="training_seeds",
        required=True,
        help='Training seeds to check (e.g., "60,61,62" or "60-69" or "60-65,67-69")',
    )
    p_missing.add_argument("--exclude", help='Seeds to exclude (e.g., "66" or "66,67")')
    p_missing.add_argument(
        "--strategy",
        choices=["all", "fullclass", "subclass", "random"],
        default="all",
        help="Filter by strategy",
    )
    p_missing.add_argument(
        "--methods",
        choices=["core", "all"],
        default="core",
        help='Methods to check: "core" (8 main methods) or "all" (includes neg_grad)',
    )
    p_missing.add_argument("--entity", help="WandB entity")
    p_missing.add_argument(
        "--prefix", default=PROJECT_PREFIX, help="Project prefix (R6, R7)"
    )
    p_missing.add_argument(
        "--verbose", "-v", action="store_true", help="Show all missing runs"
    )

    # generate-report
    p_report = subparsers.add_parser(
        "generate-report", help="Generate duplicate report (JSON)"
    )
    p_report.add_argument("--output", "-o", help="Output JSON file")
    p_report.add_argument("--entity", help="WandB entity")
    p_report.add_argument("--prefix", default=PROJECT_PREFIX, help="Project prefix")

    # -------------------------------------------------------------------------
    # ACTION COMMANDS (modify/delete)
    # -------------------------------------------------------------------------

    # cleanup-empty
    p_cleanup = subparsers.add_parser(
        "cleanup-empty", help="Delete runs with no evaluation metrics"
    )
    p_cleanup.add_argument("--project", required=True, help="WandB project name")
    p_cleanup.add_argument("--run-name", help="Filter by run name")
    p_cleanup.add_argument("--entity", help="WandB entity")
    p_cleanup.add_argument(
        "--delete", action="store_true", help="Actually delete (default: dry run)"
    )

    # delete-duplicates
    p_del_dup = subparsers.add_parser(
        "delete-duplicates", help="Delete identical duplicates"
    )
    p_del_dup.add_argument("--project", help="Specific project")
    p_del_dup.add_argument("--all", action="store_true", help="Process all projects")
    p_del_dup.add_argument("--entity", help="WandB entity")
    p_del_dup.add_argument("--prefix", default=PROJECT_PREFIX, help="Project prefix")
    p_del_dup.add_argument("--confirm", action="store_true", help="Actually delete")
    p_del_dup.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "list-projects": cmd_list_projects,
        "find-duplicates": cmd_find_duplicates,
        "find-missing": cmd_find_missing,
        "generate-report": cmd_generate_report,
        "cleanup-empty": cmd_cleanup_empty,
        "delete-duplicates": cmd_delete_duplicates,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
