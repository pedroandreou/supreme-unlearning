"""Browse the paper's published aggregate values using only Python's standard library.

Run from any working directory: python /path/to/examples/paper_results.py
This example reads existing tables; it never imports or runs the ML pipeline.
"""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import shutil


DATA_DIR = Path(__file__).resolve().parents[1] / "docs" / "results"
TABLES = ("pins_main", "pins_additional", "pins_raw")


def load_results(data_dir=DATA_DIR):
    """Read and validate the published tables without recomputing any statistics."""
    payload = {"metadata": json.loads((data_dir / "metadata.json").read_text())}
    expected_methods = {"FT", "BadT", "UNSIR", "RL", "SSD", "LFSSD"}
    for table in TABLES:
        with (data_dir / f"{table}.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        seen = set()
        for row in rows:
            key = tuple(row[field] for field in ("model", "scenario", "method"))
            if key in seen:
                raise ValueError(f"Duplicate result in {table}: {key}")
            seen.add(key)
            for field, value in row.items():
                if field not in ("model", "scenario", "method"):
                    number = float(value)
                    if not math.isfinite(number) or (
                        field.endswith("_std") and number < 0
                    ):
                        raise ValueError(f"Invalid {field} in {table}: {key}")
                    # Keep the reported decimal strings, including negative zero.
        expected = {
            (model, scenario, method)
            for model in ("ResNet18", "ViT")
            for scenario in ("fullclass", "random")
            for method in (
                expected_methods | ({"Retrain"} if table == "pins_raw" else set())
            )
            if not (scenario == "random" and method == "UNSIR")
        }
        if seen != expected:
            raise ValueError(f"Unexpected model/scenario/method coverage in {table}")
        payload[table] = rows
    return payload


def render_html(payload):
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    # JSON sits in an inert script block; escape '<' to prevent a closing tag.
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    fallback = []
    for row in payload["pins_main"]:
        values = [row[k] for k in ("model", "scenario", "method")]
        for metric in (
            "forget_accuracy_difference",
            "retain_accuracy_difference",
            "layer_distance",
        ):
            values.append(f"{row[metric + '_mean']} ± {row[metric + '_std']}")
        fallback.append(
            "<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>"
        )
    return template.replace("__PUBLISHED_DATA__", encoded).replace(
        "__FALLBACK_ROWS__", "\n".join(fallback)
    )


def export_results(payload, output):
    """Create a new, self-contained report directory; existing paths are preserved."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.html").write_text(render_html(payload), encoding="utf-8")
    (output / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for filename in ["metadata.json", *(f"{table}.csv" for table in TABLES)]:
        shutil.copyfile(DATA_DIR / filename, output / filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ResNet18", "ViT"), default="ViT")
    parser.add_argument("--scenario", choices=("fullclass", "random"), default="random")
    parser.add_argument(
        "--output",
        type=Path,
        help="Export all tables and an offline viewer to a NEW directory",
    )
    args = parser.parse_args()
    payload = load_results()
    print("SUPREME | published results | Pins Face Recognition")
    print(f"{args.model} / {args.scenario} | 10 seeds | mean ± standard deviation")
    print("Accuracy differences: unlearned minus retrained, in percentage points.\n")
    print(f"{'Method':<10}{'Forget difference':>24}{'Retain difference':>24}")
    for row in payload["pins_main"]:
        if row["model"] == args.model and row["scenario"] == args.scenario:
            forget = f"{row['forget_accuracy_difference_mean']} ± {row['forget_accuracy_difference_std']}"
            retain = f"{row['retain_accuracy_difference_mean']} ± {row['retain_accuracy_difference_std']}"
            print(f"{row['method']:<10}{forget:>24}{retain:>24}")
    print("\nSource: https://arxiv.org/abs/2606.00380, Table 1.")
    print("This example displays published summaries; it runs no experiments.")
    if args.output:
        try:
            export_results(payload, args.output)
        except FileExistsError:
            parser.error(
                f"Output already exists: {args.output}. Choose a new directory."
            )
        print(f"\nOpen in your browser: {(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
