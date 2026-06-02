<div align="center">

# Leaderboard

</div>

Community results for unlearning methods evaluated with SUPREME. To add a method,
implement and register it (see [`docs/extending.md`](../docs/extending.md)),
reproduce your numbers with a `run.sh` (see
[`methods/template/`](methods/template/)), and open a PR adding a row below.

> [!NOTE]
> [`docs/reproducing_the_paper.md`](../docs/reproducing_the_paper.md) is kept
> fixed for reproducibility. This leaderboard is the **living** table, and we'll keep
> tuning baselines and welcome community submissions. Lower is better for
> forgetting-gap metrics; higher is better for retained-set accuracy. Always state
> the dataset, model, scenario, and seed counts alongside your numbers.

## Random-sample unlearning: `Cifar10` / `ViT` (forget 1%)

<!-- Replace the placeholder rows. Columns map to SUPREME's metric families:
     Forget acc. (→ retrain), Retain acc. (↑), MIA (→ retrain), Efficiency (s). -->

| Method | Forget acc. | Retain acc. | MIA | Time (s) | Seeds (T/U/E) | Model link |
|---|---|---|---|---|---|---|
| Retrain (reference) | – | – | – | – | – | – |
| Original (no unlearning) | – | – | – | – | – | – |
| _Your method_ | – | – | – | – | – | – |

## Full-class unlearning: `<dataset>` / `<model>`

| Method | Forget acc. | Retain acc. | MIA | Time (s) | Seeds (T/U/E) | Model link |
|---|---|---|---|---|---|---|
| _Add your results_ | – | – | – | – | – | – |
