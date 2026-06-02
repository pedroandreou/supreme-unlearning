# Reproducing the paper

The SUPREME paper demonstrates the framework on **Pins Face Recognition** using **ResNet18** and **ViT**, under **full-class** and **random-sample** unlearning, across **10 training seeds** with the matched protocol (`J = K = 1`). Reproducing the reported numbers is a two-step process: run the experiment grid to populate W&B, then export the metrics and render the LaTeX tables via the analysis notebook.

## Reference version

The reported numbers were produced by the tagged reference release
[`v0.1.0-paper`](https://github.com/pedroandreou/supreme-unlearning/releases/tag/v0.1.0-paper).
To reproduce them exactly, check that tag out first:

```bash
git checkout v0.1.0-paper
```

Later commits (e.g. the pip-packaging refactor) keep the train -> unlearn ->
evaluate behaviour, defaults and seeds identical, but the tag is the
guaranteed, citable reference point.

## Prerequisites

Before running anything, make sure the environment is set up and tokens are configured. See [`docs/environment_setup.md`](environment_setup.md) for the virtual-env or Docker Dev Container instructions, the `.env` template, and the Pins Face Recognition dataset download (which requires a Kaggle key, see [`docs/adding_pinsfacerecognition.md`](adding_pinsfacerecognition.md)).

The paper's runs used a **single NVIDIA L40S GPU (48 GB VRAM)** to maintain exact numerical parity with the reference unlearning implementations. Running on a different single GPU should give numerically very close, but not bit-identical, results. Multi-GPU runs are not bit-equivalent to single-GPU runs because of distributed gradient averaging, so the paper's numbers should be reproduced single-GPU.

## 1. Run the paper's experiment grid

The command below matches the paper's setup exactly: six unlearning methods plus the retrain baseline, both scenarios, both architectures, seeds 260–269, with a 0.1% forget set for the random-sample scenario.

```bash
bash src/supreme/run_local.sh \
  --gpu 0 \
  --datasets PinsFaceRecognition \
  --models ResNet18,ViT \
  --strategies fullclass,random_ \
  --methods retrain,finetune,bad_teacher,random_labeling,unsir,ssd,lfssd \
  --forget-percs 0.001
```

The full grid above is the long-running production job. If you would rather try the pipeline first on a single cell (one method, one seed, one scenario) to confirm the environment is set up correctly before launching the full grid, use the smaller Quickstart command from the [README](../README.md#-quickstart) instead.

The pipeline writes per-stage outputs (training checkpoints, unlearning checkpoints, already-logged W&B results) to disk and to W&B, and detects and skips them on re-launch. Interruptions are therefore safe to resume: re-running the same command will pick up where it left off, only re-doing the cells that did not finish.

### What gets run

| Parameter | Value |
|---|---|
| Dataset | Pins Face Recognition (105 identities, 17,534 images) |
| Models | ResNet18 (trained from scratch on 32x32), ViT (`google/vit-base-patch16-224` fine-tuned on 224x224) |
| Unlearning methods | Retrain (baseline), Fine-Tuning (FT), Bad Teacher (BadT), Random Labels (RL), UNSIR, SSD, LFSSD |
| Scenarios | Full-class (5 identities removed), Random-sample (0.1% of training samples removed) |
| Training seeds | 260, 261, 262, 263, 264, 265, 266, 267, 268, 269 |
| Seed protocol | Matched (`J = K = 1`); unlearning and evaluation seeds collapse to the training seed |

The full-class identities (alex\_lawther, bill\_gates, danielle\_panabaker, hugh\_jackman, josh\_radnor) and the corresponding label indices are wired into `src/supreme/run_local.sh`; you do not need to set them explicitly.

For the formal seed maths and the pipeline pseudocode, see [`src/supreme/README.md`](../src/supreme/README.md) and [`docs/notation.md`](notation.md). For per-flag documentation of the underlying entrypoints, see [`docs/script_arguments.md`](script_arguments.md).

## 2. Produce the paper's LaTeX tables

After the runs finish and their metrics are logged to W&B, export the metrics and render the three paper tables.

```bash
# Export W&B run metrics and combine them into CSVs under
#   src/supreme/utils/wandb_utils/results_extraction/wandb_metrics_summary/
bash src/supreme/utils/wandb_utils/results_extraction/orchestrate_wandb_export.sh --all-existing

# Open the notebook and run all cells. It writes the three .tex files next to itself.
jupyter notebook src/supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb
```

The notebook ([`src/supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb`](../src/supreme/utils/wandb_utils/results_analysis/pins_paper_tables.ipynb)) emits three `.tex` files:

- `pins_results_table.tex` (main paper): forget/retain accuracy differences and layer-wise weight distance.
- `pins_appendix_table.tex` (appendix): forget/retain activation distances and the membership-inference-attack score difference.
- `pins_raw_values_table.tex` (appendix): per-seed raw accuracy values.

The W&B export step is described in more detail in [`docs/wandb_integration.md`](wandb_integration.md), and the metric-to-W&B-field mapping is in [`docs/wandb_fields.md`](wandb_fields.md).

## Troubleshooting

- **Pins Face Recognition is missing.** The dataset must be downloaded manually from Kaggle; follow [`docs/adding_pinsfacerecognition.md`](adding_pinsfacerecognition.md).
- **A run fails partway through.** Re-run the same `src/supreme/run_local.sh` command; completed stages are skipped. Use `--force-retraining` to force a fresh retrain if you suspect corruption.
- **The W&B export does not find your runs.** Confirm that `WANDB_KEY` is in `.env` and that the project prefix (default `R32`) matches the one your runs were logged under.
- **The notebook reports missing CSVs.** Re-run the export step; the notebook looks for CSVs under `src/supreme/utils/wandb_utils/results_extraction/wandb_metrics_summary/`.
