"""Check exported summaries against the existing paper's LaTeX table source."""

import argparse
import csv
from pathlib import Path
import re


TABLES = {
    "pins_main": "tab:pins_results_main",
    "pins_additional": "tab:pins_results_appendix",
    "pins_raw": "tab:pins_raw_values",
}


def source_rows(source, label):
    section = source.split("\\label{" + label + "}", 1)[1].split("\\end{tabular}", 1)[0]
    model = scenario = None
    rows = {}
    for line in section.splitlines():
        cells = [part.strip() for part in line.split("&")]
        if len(cells) != 15:
            continue
        if "ResNet18" in cells[0]:
            model = "ResNet18"
        elif "ViT" in cells[0]:
            model = "ViT"
        if "Full-class" in cells[1]:
            scenario = "fullclass"
        elif "Random" in cells[1]:
            scenario = "random"
        method = "Retrain" if "reference" in cells[2] else cells[2]
        numbers = []
        for integer, fraction in zip(cells[3::2], cells[4::2]):
            integer = re.sub(r"\\B\{([^}]+)\}", r"\1", integer).strip()
            fraction = re.sub(r"\\B\{([^}]+)\}", r"\1", fraction)
            fraction = fraction.replace("\\", "").strip()
            if not re.fullmatch(r"-?\d+", integer) or not re.fullmatch(
                r"\d{2}", fraction
            ):
                raise ValueError(f"Unrecognised source cell: {integer}.{fraction}")
            numbers.append(f"{integer}.{fraction}")
        key = (model, scenario, method)
        if key in rows:
            raise ValueError(f"Duplicate source row: {key}")
        rows[key] = numbers
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_tex", type=Path)
    args = parser.parse_args()
    source = args.paper_tex.read_text()
    root = Path(__file__).resolve().parents[1] / "docs" / "results"
    count = 0
    for table, label in TABLES.items():
        expected = source_rows(source, label)
        with (root / f"{table}.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            actual = {
                tuple(row[field] for field in fields[:3]): [
                    row[field] for field in fields[3:]
                ]
                for row in reader
            }
        if not expected or actual != expected:
            differences = [
                key
                for key in actual.keys() | expected.keys()
                if actual.get(key) != expected.get(key)
            ]
            raise ValueError(f"{table} differs from paper source: {differences}")
        count += len(actual) * 6
        print(f"{table}: {len(actual)} rows match the paper exactly.")
    print(f"Verified {count} published numeric values, including reported signs.")


if __name__ == "__main__":
    main()
