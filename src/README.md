# SUPREME: Algorithmic Specification

This page documents what the pipeline does at the algorithm level. For how to invoke it, see the project root [`../README.md`](../README.md) and the in-script help in [`run_local.sh`](run_local.sh), [`run_slurm.sh`](run_slurm.sh), and [`MAIN.sh`](MAIN.sh).

## Notation

All symbols used in this document are defined in [docs/notation.md](../docs/notation.md), the single source of truth across all SUPREME docs. This covers seeds ($s_t, s_u, s_e, I, J, K, \dots$), datasets ($D, D_f, D_r, \dots$), models ($M_o, M_r, M_u, M_\text{init}, \dots$), and operations ($\text{Train}, \text{Sample}, \text{Evaluate}$). Notation matches the paper (Section 2.1 and Algorithm 1).

## Seed protocols

The same pipeline supports three protocols, selected by the `--unlearning-seeds` and `--evaluation-seeds` flags in [`run_local.sh`](run_local.sh) / [`run_slurm.sh`](run_slurm.sh) (passed to [`MAIN.sh`](MAIN.sh) as the `UNLEARNING_SEEDS_J` / `EVALUATION_SEEDS_K` env vars):

| Property | Matched ($J=K=1$, default) | Decoupled ($J>1$, $K=1$) | Decoupled ($J>1$, $K>1$) |
|---|---|---|---|
| Training seed source | $s_t \leftarrow i$, $i \in \{1, \dots, I\}$ | $s_t \leftarrow i$, $i \in \{1, \dots, I\}$ | $s_t \leftarrow i$, $i \in \{1, \dots, I\}$ |
| Seed coupling | $s_t = s_u = s_e$ | $s_u$ distinct across $(s_t, j)$; $s_e = s_u$. Paper: $s_u \leftarrow (i-1) J + j$; scripts use $s_u = s_t \cdot 1000 + j$ | $s_u$ distinct across $(s_t, j)$ and $s_e$ distinct across $(s_t, j, k)$. Paper: $s_u \leftarrow (i-1) J + j$, $s_e \leftarrow (i-1) J K + (j-1) K + k$; scripts use $s_u = s_t \cdot 1000 + j$ and $s_e = s_u \cdot 1000 + k$ |
| $M_o$ trained | once per $(s_t, \text{model}, \text{dataset})$ | once per $s_t$ (shared across $J$ unlearning runs, guarded by `flock`) | once per $s_t$ (shared across $J$ unlearning runs, guarded by `flock`) |
| $M_r$ trained | once per $(s_t, c)$, seed $s_t$ | once per $(s_t, j, c)$, seed $s_u$ | once per $(s_t, j, c)$, seed $s_u$ |
| Evaluations per $(s_t, j, c, a)$ | 1 | 1 | $K$ |
| Training cost (epochs) | $I \cdot \epsilon^\text{tot}$ | $I \cdot \epsilon^\text{tot}$ | $I \cdot \epsilon^\text{tot}$ |
| Independent unlearning runs | $I \cdot \lvert C\rvert \cdot \lvert A\rvert$ | $I \cdot J \cdot \lvert C\rvert \cdot \lvert A\rvert$ | $I \cdot J \cdot \lvert C\rvert \cdot \lvert A\rvert$ |
| Independent eval results | $I \cdot \lvert C\rvert \cdot \lvert A\rvert$ | $I \cdot J \cdot \lvert C\rvert \cdot \lvert A\rvert$ | $I \cdot J \cdot K \cdot \lvert C\rvert \cdot \lvert A\rvert$ |
| Best for | dev / debug / small ablations | paper-scale statistical results | evaluator-side variability on top of $I \times J$ |

### Cost example: 10 training seeds on Cifar100 ($\epsilon^\text{tot} = 200$)

| Protocol | Training cost (epochs) | Relative |
|---|---|---|
| Matched ($J=1$) | $10 \cdot 200 = 2000$ | 100% |
| Decoupled ($J=10$) | $10 \cdot 200 = 2000$ | 100% |

Training cost is identical because $M_o$ is trained per training seed regardless of $J$. The decoupled protocol adds unlearning + retrain cost (proportional to $J$) but not training cost.
