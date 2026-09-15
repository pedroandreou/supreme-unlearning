# Acknowledgements and method citations

SUPREME builds on the Selective Synaptic Dampening and Bad Teacher codebases.
Several method implementations reimplement or adapt other published research
code. We thank the original authors and ask that you cite their papers when
using the corresponding methods, alongside the [SUPREME paper](../README.md#-citing-this-work).

## Research and code credits

| Method or implementation | Source |
|---|---|
| SSD, LFSSD | [if-loops/selective-synaptic-dampening](https://github.com/if-loops/selective-synaptic-dampening) |
| SSD-Det, LFSSD-Det | [Lanyon et al., On the importance of multiple training seeds for evaluating machine unlearning](https://arxiv.org/abs/2510.26714), Appendix B |
| Bad Teacher | [vikram2000b/bad-teaching-unlearning](https://github.com/vikram2000b/bad-teaching-unlearning) |
| UNSIR | [vikram2000b/Fast-Machine-Unlearning](https://github.com/vikram2000b/Fast-Machine-Unlearning) |
| JIT | [jwf40/Information-Theoretic-Unlearning](https://github.com/jwf40/Information-Theoretic-Unlearning) |
| SCRUB | [meghdadk/SCRUB](https://github.com/meghdadk/SCRUB) |
| NegGrad | [kklusd/Unlearning](https://github.com/kklusd/Unlearning) |

Each method's source-file header links to its original paper. For SSD-Det and
LFSSD-Det, also cite Lanyon et al.: Appendix B, "Implementation of SSD and LFSSD",
describes the corrected deterministic formulation used by these variants.

## BibTeX for the foundational codebases

```bibtex
@inproceedings{foster2024ssd,
  title     = {Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening},
  author    = {Foster, Jack and Schoepf, Stefan and Brintrup, Alexandra},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2024},
  url       = {https://arxiv.org/abs/2308.07707}
}
@inproceedings{chundawat2023badteacher,
  title     = {Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher},
  author    = {Chundawat, Vikram S and Tarun, Ayush K and Mandal, Murari and Kankanhalli, Mohan},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2023},
  url       = {https://arxiv.org/abs/2205.08096}
}
```
