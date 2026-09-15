"""The no-account example must preserve the published data and stay offline."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_results", ROOT / "examples" / "paper_results.py"
)
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def test_report_preserves_published_difference_and_reference(tmp_path):
    payload = EXAMPLE.load_results()
    row = next(
        row
        for row in payload["pins_main"]
        if (row["model"], row["scenario"], row["method"]) == ("ViT", "random", "SSD")
    )
    assert row["forget_accuracy_difference_mean"] == "-55.00"
    assert row["forget_accuracy_difference_std"] == "37.99"
    assert len([row for row in payload["pins_raw"] if row["method"] == "Retrain"]) == 4
    destination = tmp_path / "report"
    EXAMPLE.export_results(payload, destination)
    assert json.loads((destination / "results.json").read_text()) == payload
    for table in EXAMPLE.TABLES:
        assert (destination / f"{table}.csv").read_bytes() == (
            EXAMPLE.DATA_DIR / f"{table}.csv"
        ).read_bytes()
    rendered = (destination / "index.html").read_text()
    assert "__PUBLISHED_DATA__" not in rendered
    assert "__FALLBACK_ROWS__" not in rendered
    assert "<script src=" not in rendered


def test_existing_report_is_preserved(tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("existing work")
    with pytest.raises(FileExistsError):
        EXAMPLE.export_results(EXAMPLE.load_results(), tmp_path)
    assert marker.read_text() == "existing work"


def test_example_runs_without_site_packages_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "examples/paper_results.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "-55.00 ± 37.99" in result.stdout
    assert "runs no experiments" in result.stdout
