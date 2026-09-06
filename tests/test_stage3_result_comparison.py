"""Ensure hardware validation rejects score drift, not resource/timing changes."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def compare(tmp_path, distributed_score, omit_score=False):
    for case, score, elapsed in (
        ("single", 0.5, 1.0),
        ("ddp", distributed_score, 99.0),
    ):
        path = tmp_path / case
        path.mkdir()
        metrics = {"core_time_elapsed": {"final_value": elapsed}}
        if case == "single" or not omit_score:
            metrics["whole_loss"] = {"final_value": score}
        (path / "Finetune_eval_results.json").write_text(
            json.dumps({"loss": {"unlearning_method": metrics}})
        )
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/compare_stage3_results.py"),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_comparison_tolerates_roundoff_and_ignores_elapsed_time(tmp_path):
    result = compare(tmp_path, 0.500002)
    assert result.returncode == 0, result.stderr
    assert "1 scores match" in result.stdout


@pytest.mark.parametrize("score", [0.5001, float("nan"), float("inf")])
def test_comparison_rejects_drift_and_nonfinite_scores(tmp_path, score):
    assert compare(tmp_path, score).returncode != 0


def test_comparison_rejects_missing_scores(tmp_path):
    assert compare(tmp_path, 0.5, omit_score=True).returncode != 0
