# Framework comparison

SUPREME combines multi-seed image-unlearning evaluation, multi-GPU execution
and configurable numerical precision. The table summarises framework
capabilities documented in the sources below (September 2026).

| Framework | Domain in this comparison | Multi-seed | Multi-GPU | Multi-precision |
|---|---|:---:|:---:|:---:|
| [OpenUnlearning](https://github.com/locuslab/open-unlearning) | LLMs | Not shown | Yes | Yes |
| [MUBox](https://doi.org/10.1145/3734436.3734454) | Image classification | Not shown | Not shown | Not shown |
| [ERASURE](https://github.com/aiim-research/ERASURE) | Image classification | Yes | Not shown | Not shown |
| [Deep Unlearn](https://github.com/xcadet/deepunlearn) | Image classification | Yes | Not shown | Not shown |
| **SUPREME** | **Image classification** | **Yes** | **Yes** | **Yes** |

**Legend:** "Yes" indicates documented support; "Not shown" indicates support
was not identified in the cited comparison. ERASURE's entry describes its
image-classification use case within its broader domain support.

## What the columns mean

- **Multi-seed:** an integrated protocol or workflow for repeating experiments
  across random seeds. Manually rerunning a command is a different level of support.
- **Multi-GPU:** distribution of work within a pipeline stage across GPUs.
  Independently scheduling separate single-GPU experiments is a different capability.
- **Multi-precision:** selectable numerical precision in the execution workflow.
  Supported combinations depend on the device, method and backend.

SUPREME's separate training, unlearning and evaluation seeds support nested
experiments. Its registry-based design supports custom components, while
distributed execution supports repeated workloads. See
[seed protocols](seed_protocols.md) and
[implementation notes](implementation_notes.md) for configuration details.

## Sources

1. Cadet et al., *Deep Unlearn: Benchmarking Machine Unlearning for Image
   Classification*, EuroS&P 2025. [Paper](https://arxiv.org/abs/2410.01276).
2. D'Angelo et al., *How to Make Reproducible Research in Machine Unlearning with
   ERASURE*, IJCAI 2025, Demo Track.
   [Paper](https://www.ijcai.org/proceedings/2025/1255).
3. Li et al., *MUBox: A Critical Evaluation Framework of Deep Machine
   Unlearning*, SACMAT 2025. [Paper](https://doi.org/10.1145/3734436.3734454).
4. Dorna et al., *OpenUnlearning: Accelerating LLM Unlearning via Unified
   Benchmarking of Methods and Metrics*, NeurIPS Datasets and Benchmarks 2025.
   [Paper](https://arxiv.org/abs/2506.12618).
5. Andreou et al., *SUPREME: A Multi-GPU Framework for Reproducible Image
   Unlearning Method Evaluation*, WIPE-OUT 2, ECML-PKDD 2026.
   [Paper](https://arxiv.org/abs/2606.00380).

6. Andreou et al., [WIPE-OUT 2 presentation](presentations/125_Andreou_Petros.pptx),
   slide 2. Source for the capability matrix, including ERASURE's multi-seed entry.

Corrections supported by a source or a reproducible configuration are welcome
through an issue or pull request.
