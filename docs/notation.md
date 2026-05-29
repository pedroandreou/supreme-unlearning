# Notation

Single source of truth for symbols used across SUPREME's READMEs and algorithmic specifications. Matches the notation in the paper (Section 2.1 and Algorithm 1).

## Seeds and counts

| Symbol | Meaning |
|---|---|
| $I$ | number of training seeds |
| $J$ | number of unlearning seeds per training seed (default $J = 1$; paper main experiments use larger values) |
| $K$ | number of evaluation seeds per unlearning seed (default $K = 1$) |
| $i$ | training-seed index, $i \in \{1, \dots, I\}$ (paper, 1-indexed) or $\{0, \dots, I-1\}$ (codebase, 0-indexed) |
| $j$ | unlearning-seed index, $j \in \{1, \dots, J\}$ (paper) or $\{0, \dots, J-1\}$ (codebase) |
| $k$ | evaluation-seed index, $k \in \{1, \dots, K\}$ (paper) or $\{0, \dots, K-1\}$ (codebase) |
| $s_t$ | training seed; paper uses $s_t \leftarrow i$ |
| $s_u$ | unlearning seed; paper uses $s_u \leftarrow (i-1) J + j$ |
| $s_e$ | evaluation seed; paper uses $s_e \leftarrow (i-1) J K + (j-1) K + k$ |

### Independence requirement

The claim of *statistically-independent unlearning and evaluation runs* requires:

1. All $I$ training seeds are mutually distinct.
2. All $I \cdot J$ unlearning seeds, one per $(s_t, j)$ pair, are mutually distinct.
3. When $K > 1$, all $I \cdot J \cdot K$ evaluation seeds, one per $(s_t, j, k)$ triple, are mutually distinct.

The paper's formulas $s_u \leftarrow (i-1) J + j$ and $s_e \leftarrow (i-1) J K + (j-1) K + k$ satisfy these by construction. The scripts in this repo use the sparser $s_u = s_t \cdot 1000 + j$ (and $s_e = s_u \cdot 1000 + k$ when $K > 1$), which preserves independence as long as $J \leq 1000$ and $K \leq 1000$. When $J = 1$ both collapse to $s_u = s_t$; when $K = 1$ they collapse to $s_e = s_u$.

## Datasets and partitions

| Symbol | Meaning |
|---|---|
| $D, D'$ | training and test datasets |
| $D_f, D_r$ | forget and retain partitions of $D$, with $D_r := D \setminus D_f$ |
| $D'_f, D'_r$ | forget and retain partitions of $D'$ |
| $C$, $c$ | set of forget targets (classes for `fullclass`/`subclass`, ratios for `random_`); a single target $c \in C$ |
| $\tau$ | scenario type $\in \{\text{targeted}, \text{random-sample}\}$ |

## Models

| Symbol | Meaning |
|---|---|
| $M_\text{init}$ | model with initial parameters (randomly initialised or pre-trained) |
| $M_o$ | model trained on the full training set $D$ |
| $M_r$ | retrained baseline, trained from scratch on the retain set $D_r$ |
| $M_u$ | unlearned model, output of applying an unlearning method to $M_o$ |

## Unlearning methods and evaluation

| Symbol | Meaning |
|---|---|
| $A$, $a$ | set of unlearning methods; a single method $a \in A$ |
| $E$ | set of evaluation metrics |
| $P$ | set of devices (GPUs) |
| $\epsilon^\text{tot}$ | total training epochs per training run (repo-specific; e.g., 200 for Cifar100, 40 for Cifar20) |

## Operations

| Symbol | Meaning |
|---|---|
| $\text{Train}(D, s)$ | train from scratch on $D$, seeded by $s$, full schedule |
| $\text{Sample}(D, c, s)$ | uniformly sample a size-$\lceil c\lvert D\rvert \rceil$ subset of $D$ using seed $s$ (random-sample scenario only) |
| $a(M_o, D_f, D_r, s)$ | apply unlearning method $a$ to $M_o$, seeded by $s$ |
| $\text{Evaluate}(M_u, M_r, D'_f, D'_r, E, s)$ | compute the metrics in $E$ on $M_u$ against $M_r$, seeded by $s$ |
