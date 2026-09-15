# Published results

Download and explore the existing results from
[the SUPREME paper](https://arxiv.org/abs/2606.00380). This package transcribes
the reported tables at their original two-decimal precision. It contains
aggregate summaries, including the retrained reference, rather than individual
seed records or model checkpoints.

**[Open the interactive viewer](https://pedroandreou.github.io/supreme-unlearning-page/results/)**
or run the local example with Python 3.9 or later:

```bash
git clone https://github.com/pedroandreou/supreme-unlearning.git
cd supreme-unlearning
python3 examples/paper_results.py
```

The example needs only Python's standard library. It reads the existing tables
without importing PyTorch, downloading data, logging into an account, or running
training, unlearning or evaluation.

## Export an offline viewer

```bash
python3 examples/paper_results.py --output paper-results
```

Open `paper-results/index.html` in a browser. It contains the data and chart code
and works without a server or internet connection. The directory also contains
three CSV tables, metadata, and all values together in `results.json`. Choose a
new output directory for subsequent exports; existing directories are preserved.

Filter the terminal display:

```bash
python3 examples/paper_results.py --model ResNet18 --scenario fullclass
```

## Data and provenance

| File | Source in the paper | Contents |
|---|---|---|
| [pins_main.csv](pins_main.csv) | Main Table 1, `tab:pins_results_main` | Forget/retain accuracy differences and layer-wise weight distances |
| [pins_additional.csv](pins_additional.csv) | Appendix, `tab:pins_results_appendix` | Forget/retain activation distances and MIA score differences |
| [pins_raw.csv](pins_raw.csv) | Appendix, `tab:pins_raw_values` | Raw accuracy and MIA summaries, including retraining |
| [metadata.json](metadata.json) | Experimental methodology and table captions | Units, seeds, aggregation, hardware and interpretation |

The main table is also displayed on the existing project page. Values were
cross-checked against the paper's LaTeX table source. Published differences are
preserved directly: subtracting rounded raw means can differ by 0.01, and the
standard deviation of a difference cannot be recovered from two marginal
standard deviations without their covariance. Negative zero in the source is
preserved in CSV.

## Interpretation

- **Dataset:** Pins Face Recognition. Models: ResNet18 and ViT.
- **Seeds:** 260–269, with the matched protocol (J = K = 1).
- **Full-class:** average over five forget identities within each seed, then
  report mean and standard deviation across seeds.
- **Random-sample:** a 0.1% training forget set. Forget evaluation uses these
  training samples; retain evaluation uses the test set.
- **Accuracy:** raw values are percentages; differences are percentage points,
  calculated as unlearned minus retrained. Closer to zero indicates agreement
  with retraining on the selected difference metric.
- **Uncertainty:** whiskers show one standard deviation, not confidence
  intervals or individual runs. They can extend outside a metric's physical range.
  Matched stage seeds combine sources of variation; these aggregate tables do
  not identify separate stage components. See [seed protocols](../seed_protocols.md).
- **Scope:** one NVIDIA L40S GPU. These values provide no measured GPU scaling
  speedup or universal method ranking. UNSIR is excluded from random-sample
  unlearning by design. Execution times and Original-model values are absent
  from these tables, so they are not filled in here.

To run the actual experiments, follow
[Reproducing the paper](../reproducing_the_paper.md) at the reference release
`v0.1.0-paper`. To use your own components, follow
[Extending SUPREME](../extending.md).
