# Framework comparison

This is a text adaptation of **slide 2** of the
[WIPE-OUT 2 presentation](presentations/125_Andreou_Petros.pptx), added to this
repository on 9 September 2026. It compares the capabilities selected for that
presentation. It is not an independent performance benchmark or an exhaustive
inventory of each framework's current features.

| Framework | Domain in this comparison | Multi-seed | Multi-GPU | Multi-precision |
|---|---|:---:|:---:|:---:|
| [OpenUnlearning](https://github.com/locuslab/open-unlearning) | LLMs | Not shown | Yes | Yes |
| [MUBox](https://doi.org/10.1145/3734436.3734454) | Image classification | Not shown | Not shown | Not shown |
| [ERASURE](https://github.com/aiim-research/ERASURE) | Image classification | Yes | Not shown | Not shown |
| [Deep Unlearn](https://github.com/xcadet/deepunlearn) | Image classification | Yes | Not shown | Not shown |
| **SUPREME** | **Image classification** | **Yes** | **Yes** | **Yes** |

**Legend:** Yes reproduces a check on slide 2. "Not shown" reproduces a cross and
means support was not identified in that comparison, not that a user could
never implement or configure it. For ERASURE, the domain is the image-classification
use case in the slide, not a claim that the framework is confined to images.

## What the columns mean

- **Multi-seed:** an integrated protocol or workflow for repeating experiments
  across random seeds. Manually rerunning a command is a different level of support.
- **Multi-GPU:** distribution of work within a pipeline stage across GPUs.
  Independently scheduling separate single-GPU experiments is a different capability.
- **Multi-precision:** selectable numerical precision in the execution workflow.
  Supported combinations depend on the device, method and backend.

SUPREME supports separate training, unlearning and evaluation seeds. The paper's
demonstration uses the matched setting (J = K = 1), and its reported results
come from one GPU. The comparison does not measure speedups, GPU-hours,
numerical equivalence across precisions, privacy guarantees or ease of installation.
See [implementation notes](implementation_notes.md) for method/backend restrictions.

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

The presentation credits ERASURE with multi-seed support; the paper's introductory
comparison describes only Deep Unlearn as supporting it. This table follows the
presentation. Corrections supported by a source or a reproducible configuration
are welcome through an issue or pull request.
