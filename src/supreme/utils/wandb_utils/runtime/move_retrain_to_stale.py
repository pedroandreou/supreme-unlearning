#!/usr/bin/env python3
"""
Move retrain WandB runs from M32 projects to M32_stale projects.

Finds all retrain runs for the specified training seeds across all scaled-experiment
projects, recreates them in the corresponding M32_stale project (preserving config,
summary, tags, and notes), then deletes the originals.

WandB 0.24.x has no direct run.move() - this script implements copy-then-delete.
Projects are processed in parallel for speed.

Usage:
    # Preview (no changes):
    python move_retrain_to_stale.py --training-seeds 1,2,3 --dry-run

    # Actually move:
    python move_retrain_to_stale.py --training-seeds 1,2,3 --move

    # Control parallelism:
    python move_retrain_to_stale.py --training-seeds 1,2,3 --move --workers 8
"""

import argparse
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import wandb
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
except ImportError:
    print(
        "Error: Required packages not installed. Run: pip install wandb python-dotenv"
    )
    sys.exit(1)


PRECISION = "32-true"
SRC_PREFIX = "M32"
DST_PREFIX = "M32_stale"

SCALED_COMBOS = {
    "fullclass": {
        "Cifar100": ["rocket", "mushroom", "baby", "lamp", "sea"],
        "Cifar20": [
            "vehicle2",
            "veg",
            "people",
            "electrical_devices",
            "natural_scenes",
        ],
    },
    "subclass": {
        "Cifar20": ["rocket", "mushroom", "baby", "lamp", "sea"],
    },
}

# Unlearning seeds: actual seed = training_seed * 1000 + j, j in 0..9
UNLEARNING_SEED_INDICES = list(range(10))

_print_lock = threading.Lock()


def tprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def project_names(prefix: str, model: str = "ResNet18") -> list[str]:
    names = []
    for strategy, datasets in SCALED_COMBOS.items():
        for dataset, targets in datasets.items():
            for target in targets:
                names.append(
                    f"{prefix}_UNLEARNING_{model}_{dataset}_{strategy}_{target}_precision_{PRECISION}"
                )
    return names


def build_run_names(training_seeds: list[int], methods: list[str]) -> list[str]:
    names = []
    for method in methods:
        for ts in training_seeds:
            for j in UNLEARNING_SEED_INDICES:
                useed = ts * 1000 + j
                names.append(f"{method}_tseed{ts}_useed{useed}")
    return names


def build_run_suffixes(training_seeds: list[int]) -> set[str]:
    """Suffixes used when --methods=all: match any '<anything>_tseed{ts}_useed{useed}'."""
    suffixes = set()
    for ts in training_seeds:
        for j in UNLEARNING_SEED_INDICES:
            useed = ts * 1000 + j
            suffixes.add(f"_tseed{ts}_useed{useed}")
    return suffixes


def find_runs(
    api, entity: str, project: str, run_names: set[str], match_suffixes: set[str] = None
) -> list:
    try:
        all_runs = api.runs(f"{entity}/{project}", per_page=200)
        if match_suffixes:
            return [
                r
                for r in all_runs
                if any(r.display_name.endswith(sfx) for sfx in match_suffixes)
            ]
        return [r for r in all_runs if r.display_name in run_names]
    except wandb.errors.CommError:
        return []
    except ValueError:
        return []
    except Exception as e:
        tprint(f"  Warning: error querying {project}: {e}")
        return []


def to_plain(v):
    """Recursively convert WandB internal types (SummarySubDict etc.) to plain Python dicts."""
    if hasattr(v, "items"):
        return {k: to_plain(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [to_plain(item) for item in v]
    return v


def copy_run_to_stale(entity: str, src_run, dst_project: str, dry_run: bool) -> bool:
    """Recreate src_run in dst_project with config + summary + tags. Returns True on success."""
    if dry_run:
        tprint(f"    [DRY RUN] Would copy → {dst_project} / {src_run.display_name}")
        return True

    # Capture data from src_run BEFORE wandb.init() - wandb.init() mutates
    # global state in a way that makes later src_run.summary access return the
    # freshly-initialized destination run's (empty) summary instead.
    name = src_run.display_name
    tags = list(src_run.tags or []) + ["M32_stale"]
    config = dict(src_run.config)
    notes = (src_run.notes or "") + "\n[Archived from M32 - stale results prior to fix]"
    summary = {
        k: to_plain(v) for k, v in src_run.summary.items() if not k.startswith("_")
    }

    try:
        wandb.init(
            project=dst_project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            reinit=True,
            settings=wandb.Settings(silent=True, init_timeout=120),
        )
        if summary:
            wandb.log(summary)
        wandb.finish(quiet=True)
        tprint(f"    Copied  → {dst_project} / {name}")
        return True
    except Exception as e:
        tprint(f"    ERROR copying {name}: {e}")
        try:
            wandb.finish(quiet=True, exit_code=1)
        except Exception:
            pass
        return False


def process_project(
    entity: str,
    src_project: str,
    target_run_names: set[str],
    dry_run: bool,
    match_suffixes: set[str] = None,
):
    """Process one project: find runs, copy to stale, delete originals. Returns (found, moved, failed)."""
    # Each worker process needs its own API instance to avoid shared state issues
    api = wandb.Api()
    dst_project = src_project.replace(SRC_PREFIX, DST_PREFIX, 1)
    runs = find_runs(api, entity, src_project, target_run_names, match_suffixes)

    if not runs:
        return 0, 0, 0

    tprint(f"  {src_project}")
    tprint(f"    Found {len(runs)} retrain run(s)")

    found = len(runs)
    moved = 0
    failed = 0

    for run in runs:
        ok = copy_run_to_stale(entity, run, dst_project, dry_run)
        if ok:
            if not dry_run:
                try:
                    run.delete()
                    tprint(f"    Deleted ← {src_project} / {run.display_name}")
                    moved += 1
                except Exception as e:
                    tprint(f"    ERROR deleting {run.display_name} from source: {e}")
                    failed += 1
            else:
                moved += 1
        else:
            failed += 1

    return found, moved, failed


def main():
    parser = argparse.ArgumentParser(
        description="Move retrain WandB runs from M32 to M32_stale"
    )
    parser.add_argument(
        "--training-seeds",
        required=True,
        help="Comma-separated training seeds (e.g. 1,2,3)",
    )
    parser.add_argument(
        "--model",
        default="ResNet18",
        help="Model name (default: ResNet18)",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="WandB entity (default: your default entity)",
    )
    parser.add_argument(
        "--methods",
        default="retrain",
        help="Comma-separated method names (e.g. retrain,original,ssd) or 'all' to match every method (default: retrain)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for project processing (default: 4)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="List runs without moving")
    mode.add_argument(
        "--move", action="store_true", help="Copy to M32_stale then delete from M32"
    )
    args = parser.parse_args()

    training_seeds = [int(s.strip()) for s in args.training_seeds.split(",")]

    api = wandb.Api()
    entity = args.entity or api.default_entity

    src_projects = project_names(SRC_PREFIX, args.model)

    if args.methods.strip().lower() == "all":
        target_run_names = set()
        match_suffixes = build_run_suffixes(training_seeds)
        method_label = "ALL methods"
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        target_run_names = set(build_run_names(training_seeds, methods))
        match_suffixes = None
        method_label = ",".join(methods)

    mode_label = "DRY RUN" if args.dry_run else "MOVE"
    print(
        f"=== {mode_label}: {method_label} runs for training seeds {training_seeds} ==="
    )
    print(f"Entity:        {entity}")
    print(f"Source prefix: {SRC_PREFIX}  →  Dest prefix: {DST_PREFIX}")
    print(f"Projects:      {len(src_projects)}")
    if target_run_names:
        print(f"Run names:     {len(target_run_names)}")
    else:
        print(
            f"Match suffixes: {len(match_suffixes)} (any method matching *_tseed{{ts}}_useed{{useed}})"
        )
    print(f"Workers:       {args.workers}")
    print()

    total_found = 0
    total_moved = 0
    total_failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_project,
                entity,
                proj,
                target_run_names,
                args.dry_run,
                match_suffixes,
            ): proj
            for proj in src_projects
        }
        for future in as_completed(futures):
            found, moved, failed = future.result()
            total_found += found
            total_moved += moved
            total_failed += failed

    print()
    verb = "would be moved" if args.dry_run else "moved"
    print(
        f"=== Found: {total_found}  |  {verb}: {total_moved}  |  Errors: {total_failed} ==="
    )
    if total_failed > 0:
        print(
            "WARNING: Some runs failed - check above. Originals were NOT deleted for failed copies."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
