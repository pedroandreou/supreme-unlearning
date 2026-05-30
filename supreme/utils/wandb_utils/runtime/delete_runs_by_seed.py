#!/usr/bin/env python3
"""
Delete WandB runs for specific methods and training seeds.

Generates the relevant project names for the scaled experiment (Cifar100 fullclass,
Cifar20 fullclass, Cifar20 subclass - ResNet18 only), then finds and deletes all
runs whose display_name matches the given method + training seed pattern.

Usage:
    # Dry run (list what would be deleted):
    python delete_runs_by_seed.py --method ssd --training-seeds 0,60 --prefix R32 --dry-run

    # Actually delete:
    python delete_runs_by_seed.py --method ssd --training-seeds 0,60 --prefix R32 --delete
"""

import argparse
import sys
import os

try:
    import wandb
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv())
except ImportError:
    print("Error: Required packages not installed. Run: pip install wandb python-dotenv")
    sys.exit(1)


PRECISION = "32-true"

SCALED_COMBOS = {
    "fullclass": {
        "Cifar100": ["rocket", "mushroom", "baby", "lamp", "sea"],
        "Cifar20": ["vehicle2", "veg", "people", "electrical_devices", "natural_scenes"],
    },
    "subclass": {
        "Cifar20": ["rocket", "mushroom", "baby", "lamp", "sea"],
    },
}


def generate_project_names(prefix: str, model: str = "ResNet18") -> list[str]:
    """Generate project names for the scaled experiment combos."""
    projects = []
    for strategy, datasets in SCALED_COMBOS.items():
        for dataset, targets in datasets.items():
            for target in targets:
                projects.append(
                    f"{prefix}_UNLEARNING_{model}_{dataset}_{strategy}_{target}_precision_{PRECISION}"
                )
    return projects


def find_runs_by_display_name(api, entity: str, project: str, display_name: str):
    """Query wandb for runs matching an exact display_name."""
    try:
        return list(api.runs(
            f"{entity}/{project}",
            filters={"display_name": display_name},
        ))
    except wandb.errors.CommError:
        return []
    except Exception as e:
        print(f"    Warning: error querying {project}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Delete WandB runs by method and training seed")
    parser.add_argument("--method", required=True, help="Unlearning method name (e.g. ssd)")
    parser.add_argument("--training-seeds", required=True, help="Comma-separated training seeds (e.g. 0,60)")
    parser.add_argument("--unlearning-seeds", default="0", help="Comma-separated unlearning seeds to check (default: 0)")
    parser.add_argument("--prefix", default="R32", help="WandB project prefix (default: R32)")
    parser.add_argument("--model", default="ResNet18", help="Model name (default: ResNet18)")
    parser.add_argument("--entity", default=None, help="WandB entity (default: your default entity)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="List runs without deleting")
    group.add_argument("--delete", action="store_true", help="Actually delete matching runs")
    args = parser.parse_args()

    training_seeds = [int(s.strip()) for s in args.training_seeds.split(",")]
    unlearning_seeds = [int(s.strip()) for s in args.unlearning_seeds.split(",")]
    projects = generate_project_names(args.prefix, args.model)

    api = wandb.Api()
    entity = args.entity or api.default_entity

    # Build exact run names to search for (both naming conventions)
    target_run_names = []
    for ts in training_seeds:
        for us in unlearning_seeds:
            target_run_names.append(f"{args.method}_tseed{ts}_useed{us}")
            target_run_names.append(f"{args.method}_seed{us}")

    mode_label = "DRY RUN" if args.dry_run else "DELETE"
    print(f"=== {mode_label}: {args.method} for training seeds {training_seeds} ===")
    print(f"Entity: {entity}")
    print(f"Projects to scan: {len(projects)}")
    print(f"Run names to match: {target_run_names}")
    print()

    total_deleted = 0
    for project in projects:
        project_count = 0
        for run_name in target_run_names:
            runs = find_runs_by_display_name(api, entity, project, run_name)
            for run in runs:
                if args.dry_run:
                    print(f"  [DRY RUN] Would delete: {run.name}  (project={project}, id={run.id}, state={run.state})")
                else:
                    try:
                        run.delete()
                        print(f"  Deleted: {run.name}  (project={project}, id={run.id})")
                    except Exception as e:
                        print(f"  ERROR deleting {run.name}: {e}")
                project_count += len(runs)

        if project_count > 0:
            print(f"  -> {project}: {project_count} run(s)")
            print()
        total_deleted += project_count

    print(f"=== Total: {total_deleted} run(s) {'would be deleted' if args.dry_run else 'deleted'} ===")


if __name__ == "__main__":
    main()
