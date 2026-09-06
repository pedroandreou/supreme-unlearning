"""Compare numerical Stage 3 scores, excluding timing/resource measurements."""

import argparse
import json
import math
from pathlib import Path


def numerical_scores(value, path=()):
    scores = {}
    if not isinstance(value, dict):
        return scores
    if "final_value" in value and "unlearning_method" in path:
        if "core_time_elapsed" not in path:
            score = float(value["final_value"])
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite score at {'/'.join(path)}")
            scores["/".join(path)] = score
    for key, child in value.items():
        scores.update(numerical_scores(child, (*path, key)))
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    results = {}
    for case in sorted(args.directory.iterdir()):
        if not case.is_dir():
            continue
        files = list(case.rglob("*_eval_results.json"))
        if len(files) != 1:
            raise ValueError(
                f"Expected one evaluation JSON in {case}, found {len(files)}"
            )
        results[case.name] = json.loads(files[0].read_text())
    baseline = numerical_scores(results["single"])
    if not baseline or len(results) < 2:
        raise ValueError(
            "Comparison requires nonempty single-GPU and distributed scores"
        )
    for case, result in results.items():
        scores = numerical_scores(result)
        if scores.keys() != baseline.keys():
            raise AssertionError(f"Metric keys differ for {case}")
        for key, expected in baseline.items():
            if not math.isclose(scores[key], expected, rel_tol=0, abs_tol=args.atol):
                raise AssertionError(f"{case} {key}: {scores[key]} != {expected}")
        delta = max(abs(scores[key] - expected) for key, expected in baseline.items())
        print(
            f"{case}: {len(scores)} scores match; max absolute difference={delta:.12g}"
        )
        print(f"  execution_config={result.get('execution_config')}")


if __name__ == "__main__":
    main()
