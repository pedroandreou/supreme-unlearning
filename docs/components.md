# Available components

Registry-based components are **user-extensible** - implement the relevant interface and register the module path, either in-tree or **from your own package** (runtime API or packaging entry points, no edits to SUPREME). See [`docs/extending.md`](extending.md). The components provided via Lightning Fabric cover the supported hardware and execution configurations.

## Registry-based (user-extensible)

| Component | Available implementations |
|---|---|
| **Datasets** | [CIFAR-10](../src/supreme/datasets/datasets.py), [CIFAR-20](../src/supreme/datasets/datasets.py), [CIFAR-100](../src/supreme/datasets/datasets.py), [PinsFaceRecognition](../src/supreme/datasets/datasets.py), [Caltech-101](../src/supreme/datasets/datasets.py) |
| **Models** | [ResNet18](../src/supreme/models/ResNet18.py), [Vision Transformer (ViT)](../src/supreme/models/ViT.py) |
| **Baseline & reference** | [Retrain](../src/supreme/methods/baselines/retrain.py) (gold-standard baseline), [Original](../src/supreme/methods/baselines/original.py) (unmodified reference) |
| **Unlearning methods** | [Fine-Tuning (FT)](../src/supreme/methods/unlearning_methods/finetune.py), [Bad Teacher (BadT)](../src/supreme/methods/unlearning_methods/bad_teacher.py), [Random Labels (RL)](../src/supreme/methods/unlearning_methods/random_labeling.py), [UNSIR](../src/supreme/methods/unlearning_methods/unsir.py), [SSD](../src/supreme/methods/unlearning_methods/ssd.py), [LFSSD](../src/supreme/methods/unlearning_methods/lfssd.py), [SSD-Det](../src/supreme/methods/unlearning_methods/ssd_det.py), [LFSSD-Det](../src/supreme/methods/unlearning_methods/lfssd_det.py), [ASSD](../src/supreme/methods/unlearning_methods/assd.py), [SCRUB](../src/supreme/methods/unlearning_methods/scrub.py), [JIT](../src/supreme/methods/unlearning_methods/jit.py) |
| **Evaluation metrics** | [Accuracy](../src/supreme/eval_metrics/accuracy.py), [Loss/Error](../src/supreme/utils/training/training_utils.py), [ZRF](../src/supreme/eval_metrics/zrf.py), [Activation Distance](../src/supreme/eval_metrics/activation_distance.py), [JS-Divergence](../src/supreme/eval_metrics/jsdiv.py), [Layer-wise Distance](../src/supreme/eval_metrics/layerwise_distance.py), [Membership Inference Attack](../src/supreme/eval_metrics/membership_inference_attack.py), [Completeness](../src/supreme/eval_metrics/completeness.py), [Resource Consumption](../src/supreme/eval_metrics/resource_consumption.py), [Time](../src/supreme/eval_metrics/time.py) |
| **Unlearning scenarios** | Full-class, Subclass, Random sample |

## Provided via Lightning Fabric

| Component | Available implementations |
|---|---|
| **Accelerators** | [CPU](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.CPUAccelerator.html), [CUDA](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.CUDAAccelerator.html), [MPS](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.MPSAccelerator.html), [TPU](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.accelerators.XLAAccelerator.html) |
| **Precision modes** | [64-true, 32-true, 16-mixed, bf16-mixed, 16-true, bf16-true](https://lightning.ai/docs/fabric/2.1.0/fundamentals/precision.html), [transformer-engine, transformer-engine-float16](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.plugins.precision.TransformerEnginePrecision.html) (FP8), [nf4, nf4-dq, fp4, fp4-dq, int8, int8-training](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.plugins.precision.BitsandbytesPrecision.html) |
| **Distributed strategies** | [DDP](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.DDPStrategy.html), [FSDP](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.FSDPStrategy.html), [DeepSpeed](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.strategies.DeepSpeedStrategy.html) (ZeRO Stage 1/2/3) |
| **Loggers** | [Weights & Biases](https://docs.wandb.ai/guides/integrations/lightning), [TensorBoard](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.loggers.TensorBoardLogger.html), [CSV](https://lightning.ai/docs/fabric/2.1.0/api/generated/lightning.fabric.loggers.CSVLogger.html) |
