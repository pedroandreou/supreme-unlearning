# Future Work

## Gradient Accumulation for Multi-GPU Efficiency

When scaling to multi-GPU setups, the current batch size scaling approach (dividing batch size by number of GPUs) can lead to very small per-GPU batch sizes, which may cause performance degradation due to underutilized GPU compute capacity.

A promising alternative is **gradient accumulation**, which allows simulating larger effective batch sizes without increasing memory usage. Instead of synchronizing gradients after every micro-batch, gradients are accumulated over multiple forward/backward passes before performing a single optimizer step.

**Benefits:**
- Maintains larger per-GPU batch sizes for better GPU utilization
- Reduces communication overhead by synchronizing less frequently
- Can achieve the same effective batch size with more flexibility

**Resources:**
- [Accelerate: Gradient Synchronization](https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization) - Useful for understanding the concept and trade-offs (even though we use Lightning Fabric, Accelerate's documentation provides clear explanations)
- [Lightning Fabric: Gradient Accumulation](https://lightning.ai/docs/fabric/2.4.0/advanced/gradient_accumulation.html) - Implementation guide for our framework

**Implementation Considerations:**
- Requires careful handling of gradient synchronization timing with `fabric.no_backward_sync()`
- Learning rate scheduling may need adjustment when using accumulation
- Some unlearning methods may require specific adaptations
