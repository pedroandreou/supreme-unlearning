# <METHOD NAME>

- Paper title, authors, and links (paper / code).

Provide a concise summary of your method and its contributions. Please avoid
embedding large images to keep the repository size manageable.

## Availability

- [ ] **Plugin package** (recommended): name + install command (e.g. `pip install my-unlearning-method`) and the registration call / entry point it exposes.
- [ ] **In-tree**: link to the PR / files under `supreme/`.

## Setup

Please describe what's needed to reproduce your results:

- [ ] **Hyperparameters & search space:** key hyperparameters, their ranges, and the number of trials.
- [ ] **Scenario(s):** full-class / subclass / random-sample, and the forget percentage(s) for random-sample.
- [ ] **Datasets & models:** which of SUPREME's (or your own) datasets and architectures you ran on.
- [ ] **Seeds:** training / unlearning / evaluation seed counts used.
- [ ] **Compute:** accelerator (CUDA + GPU model / MPS / TPU), number of GPUs, and any distributed strategy (DDP / FSDP / DeepSpeed).
- [ ] **Other details:** anything else crucial for reproduction.

## Results

Provide a well-documented [`run.sh`](run.sh) containing every command needed to
reproduce your final numbers. Report against SUPREME's metric families:
forgetting, utility, privacy, behavioural/parametric equivalence, and efficiency.

If you can, upload the final unlearned model(s) to HuggingFace and link them
here so results can be re-evaluated as metrics evolve. Don't forget to add a row
to the [leaderboard](../../leaderboard.md).

## Citation

If you use this work, please cite:

```bibtex
<YOUR CITATION bibtex>

@misc{supreme2026,
  title  = {SUPREME: A Multi-GPU Framework for Reproducible Image Unlearning Method Evaluation},
  author = {Petros Andreou, Jamie Lanyon, Axel Finke, Georgina Cosma},
  year   = {2026},
  eprint = {2606.00380},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2606.00380}
}
```
