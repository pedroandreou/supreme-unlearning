# Training, unlearning and evaluation seeds

## Why the distinction matters

Ten repetitions of unlearning from one checkpoint are not ten independently
trained starting models. The former describe one model's conditional behaviour;
the latter sample different starting models.

[Lanyon et al., Section 2.2](https://arxiv.org/html/2510.26714v5#S2.SS2)
separate training and unlearning variance. For a metric with deterministic
evaluation, their Equation (4) gives

$$
\operatorname{Var}(\bar Z)
= \frac{\sigma_{\mathrm{train}}^2}{I}
+ \frac{\sigma_{\mathrm{unlearn}}^2}{IJ}.
$$

Here, $I$ counts independently trained starting models and $J$ counts unlearning
repetitions per starting model. Under the paper's sampling assumptions,
increasing $J$ reduces only the second term. If training variance is positive,
holding $I$ fixed leaves a nonzero uncertainty floor. This is a statement about
the variance of the estimated mean, not a claim that every extra unlearning run
is useless or that a universal seed count is sufficient.

## What SUPREME controls

The [pipeline](../src/supreme/README.md) exposes three nested levels:

| Level | Repeated operation | Interpretation |
|---|---|---|
| Training, $I$ | Train an original model for each training seed | Sample variation between starting models |
| Unlearning, $J$ | Reuse each original model across unlearning seeds | Sample variation within that starting model |
| Evaluation, $K$ | Re-evaluate each unlearned model across evaluation seeds | Investigate randomness in a stochastic evaluator |

SUPREME schedules these repetitions and retains their seed identities. Analysis
must preserve this grouping: the $J$ runs sharing a checkpoint are not $J$
independent training replicates. Separate stage controls enable a variance
analysis; the launchers do not automatically estimate variance components.

For a stochastic evaluator, applying total variance again separates
between-training, within-training/between-unlearning, and within-model evaluation
components. This is an extension of the nested design, not a three-component
result reported by Jamie's paper, which assumes deterministic metric evaluation
in its theoretical analysis.

Stage-level variation is also not necessarily algorithm-only variation. In the
current pipeline, the unlearning seed controls the retrained reference and, for
random-sample unlearning, the forget-set draw. Their variation can therefore
enter that stage's component. A finer attribution requires a correspondingly
controlled design.

## Select a protocol

After following the [environment setup](environment_setup.md), the local and
SLURM launchers accept these flags. They configure real experiments, unlike the
[published-results example](results/README.md), which only reads existing tables.

| Design | Launcher flags | What it supports |
|---|---|---|
| Matched | `--training-seeds 260,261,262 --unlearning-seeds "0" --evaluation-seeds "0"` | Repeated full pipelines; stage effects remain combined |
| Nested training/unlearning | `--training-seeds 260,261,262 --unlearning-seeds "0 1 2" --evaluation-seeds "0"` | Within-model and between-model comparisons |
| Nested evaluation | `--training-seeds 260,261,262 --unlearning-seeds "0 1 2" --evaluation-seeds "0 1 2"` | Additional within-model evaluator repetitions |

These small counts illustrate syntax, not a recommended experimental budget.
Use distinct training seeds and distinct nonnegative inner indices below 1000;
see [seed mapping](notation.md#independence-requirement). Report $I$, $J$ and $K$
separately, alongside the data split, method configuration and hardware.

Choose repetitions according to the metric, stage variability and cost. See
[Lanyon et al., Section 2.3](https://arxiv.org/html/2510.26714v5#S2.SS3)
for allocation guidance rather than assuming that more unlearning seeds always
offer the best use of compute. A deterministic method and deterministic metric
do not gain additional variation from changing only the unlearning seed.

## Keep the two papers distinct

The [training-seed study](https://arxiv.org/abs/2510.26714v5) reports a
training/unlearning variance analysis using 75 training seeds and ten unlearning
seeds per training seed in its image experiments. SUPREME supported the extended
image-classification experiments; this does not make its LLM or federated
learning experiments SUPREME capabilities.

The [SUPREME framework paper](https://arxiv.org/abs/2606.00380) demonstrates
Pins Face Recognition with ten training seeds and matched $J=K=1$. Its
[published tables](results/README.md) show aggregate spread across those
pipelines. They do not contain the nested observations needed to recover
separate training, unlearning and evaluation variance components.
