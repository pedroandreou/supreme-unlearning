"""Render shareable graphics from the existing paper values and slide-2 comparison."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INK = "#201c2c"
PURPLE = "#582c83"
MUTED = "#625d70"
BACKGROUND = "#faf8fc"


def save(fig, name):
    for extension in ("png", "svg", "pdf"):
        output = ROOT / "assets" / f"{name}.{extension}"
        fig.savefig(
            output,
            dpi=100,
            facecolor=fig.get_facecolor(),
        )
        if extension == "svg":
            # Matplotlib emits trailing spaces in SVG paths; keep diffs clean.
            output.write_text(
                "\n".join(line.rstrip() for line in output.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def seed_variation():
    with (ROOT / "docs/results/pins_main.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.27, top=0.70, wspace=0.27)
    fig.text(
        0.055,
        0.93,
        "SUPREME  /  PUBLISHED RESULTS",
        color=PURPLE,
        size=11,
        weight="bold",
    )
    fig.text(
        0.055,
        0.85,
        "Published variation across ten seeds",
        color=INK,
        size=25,
        weight="bold",
    )
    fig.text(
        0.055,
        0.79,
        "Pins Face Recognition · random-sample unlearning · forget 0.1%",
        color=MUTED,
        size=12,
    )
    for ax, model in zip(axes, ("ResNet18", "ViT")):
        subset = [r for r in rows if r["model"] == model and r["scenario"] == "random"]
        means = [float(r["forget_accuracy_difference_mean"]) for r in subset]
        stds = [float(r["forget_accuracy_difference_std"]) for r in subset]
        ax.set_facecolor(BACKGROUND)
        ax.errorbar(
            means,
            range(len(subset)),
            xerr=stds,
            fmt="o",
            color=PURPLE,
            capsize=5,
            lw=2,
            ms=7,
        )
        ax.set_yticks(range(len(subset)), [r["method"] for r in subset], color=INK)
        ax.invert_yaxis()
        ax.set_ylim(len(subset) - 0.5, -0.5)
        ax.set_xlim(-110, 40)
        ax.set_xticks([-100, -75, -50, -25, 0, 25])
        ax.axvline(0, color="#9c8ca9", ls="--", lw=1)
        ax.grid(axis="x", color="#e7e0ed", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(model, color=INK, size=14, weight="bold", pad=12)
        ax.set_xlabel(
            "Forget accuracy difference (percentage points)",
            color=MUTED,
            size=10,
            labelpad=12,
        )
        ax.tick_params(axis="both", length=0, labelsize=11, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.055,
        0.145,
        "Dots: mean. Whiskers: one standard deviation across seeds (not confidence intervals).",
        color=INK,
        size=11,
    )
    fig.text(
        0.055,
        0.102,
        "Difference = unlearned minus retrained. Zero indicates agreement on this metric.",
        color=MUTED,
        size=10,
    )
    fig.text(
        0.055,
        0.057,
        "Existing Table 1 · arxiv.org/abs/2606.00380 · Single NVIDIA L40S GPU",
        color=MUTED,
        size=10,
    )
    save(fig, "published-seed-variation")


def framework_comparison():
    fig, ax = plt.subplots(figsize=(12, 6.3), facecolor=BACKGROUND)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    ax.text(
        0.055,
        0.93,
        "SUPREME  /  FRAMEWORK COMPARISON",
        color=PURPLE,
        size=11,
        weight="bold",
    )
    ax.text(
        0.055,
        0.85,
        "Image unlearning, across seeds and devices",
        color=INK,
        size=24,
        weight="bold",
    )
    ax.text(
        0.055,
        0.79,
        "Capabilities compared in the WIPE-OUT 2 presentation · September 2026",
        color=MUTED,
        size=12,
    )
    rows = [
        ["OpenUnlearning", "LLMs", "Not shown", "Yes", "Yes"],
        ["MUBox", "Image classification", "Not shown", "Not shown", "Not shown"],
        ["ERASURE", "Image classification", "Yes", "Not shown", "Not shown"],
        ["Deep Unlearn", "Image classification", "Yes", "Not shown", "Not shown"],
        ["SUPREME", "Image classification", "Yes", "Yes", "Yes"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Framework", "Domain", "Multi-seed", "Multi-GPU", "Multi-precision"],
        colWidths=[0.20, 0.27, 0.17, 0.17, 0.19],
        cellLoc="center",
        bbox=[0.055, 0.265, 0.89, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#ded8e5")
        cell.set_linewidth(0.5)
        cell.set_facecolor("white" if row % 2 else "#f2edf7")
        cell.set_text_props(color=INK)
        if row == 0:
            cell.set_facecolor(PURPLE)
            cell.set_text_props(color="white", weight="bold")
        if row == 5:
            cell.set_facecolor("#e9def3")
            cell.set_text_props(color=PURPLE, weight="bold")
    ax.text(
        0.055,
        0.18,
        "“Not shown” means support was not identified in slide 2; it does not mean impossible.",
        color=INK,
        size=11,
    )
    ax.text(
        0.055,
        0.13,
        "Capability comparison, not measured speed or unlearning quality. Definitions and sources:",
        color=MUTED,
        size=10,
    )
    ax.text(
        0.055,
        0.077,
        "github.com/pedroandreou/supreme-unlearning/blob/main/docs/framework_comparison.md",
        color=PURPLE,
        size=10,
    )
    save(fig, "framework-comparison")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    seed_variation()
    framework_comparison()
