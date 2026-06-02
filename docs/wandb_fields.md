# W&B Metrics Field Documentation

This document contains the W&B field naming conventions, paper-to-WandB mappings, and complete per-metric field paths for SUPREME. Field names are defined in [metrics_main.py](../supreme/eval_metrics/metrics_main.py).

## Evaluation Resource Tracking Flag

By default, SUPREME tracks resource consumption (time/memory/power) for **unlearning methods** but **NOT** for evaluation metrics computation. This is controlled by the `track_evaluation_resources` flag in [run_local.sh](../supreme/run_local.sh):

```bash
track_evaluation_resources=false  # Default: disabled
```

- **When enabled (`true`):** All `eval_metric.*` fields will contain resource tracking data
- **When disabled (`false`, default):** All `eval_metric.*` fields will be `None`; only metric values and unlearning method resources are tracked

## Field Naming Conventions

SUPREME uses a hierarchical naming structure with two patterns:

**Pattern 1 - Data-based metrics** (evaluated on test sets): `<metric_name>.test.<category>.<field>`
- Used by: `accuracy`, `loss`, `zrf`, `jsdiv`, `membership_inference_attack`, `activation_distance`, `completeness`

**Pattern 2 - Parameter-based metrics** (no test data needed): `<metric_name>.<category>.<field>`
- Used by: `layerwise_distance`, `time`

**Category distinction:**
- `unlearning_method` - contains metric results AND resources consumed BY the unlearning process
- `eval_metric` - contains ONLY resources consumed FOR computing the metric

**Example:** `accuracy.test.unlearning_method.core_time_elapsed` = time taken BY the unlearning method. `accuracy.test.eval_metric.core_time_elapsed` = time taken FOR computing accuracy.

## Paper Metrics to WandB Mapping

**Legend:** M<sub>u</sub> = unlearned model, M<sub>r</sub> = retrained model, D<sub>f</sub> = forget set, D<sub>r</sub> = retain set, D<sub>test</sub> = whole test set

#### Representation Metrics

| Paper Metric | WandB Key Path | Optimal | Description |
|--------------|----------------|---------|-------------|
| Activation Distance | `activation_distance.test.unlearning_method.activation_distance_forget.final_value` | Min → 0 | L2 distance between M<sub>u</sub> and M<sub>r</sub> final-layer outputs on D<sub>f</sub> (also logged as `_whole` and `_retain`) |
| JS-Divergence | `jsdiv.test.unlearning_method.jsdiv_forget.final_value` | Min → 0 | Jensen-Shannon divergence between M<sub>u</sub> and M<sub>r</sub> output distributions on D<sub>f</sub> (also logged as `_whole` and `_retain`) |
| Layer-wise Distance | `layerwise_distance.unlearning_method.layerwise_distance.final_value` | Min → 0 | Euclidean distance between M<sub>u</sub> and M<sub>r</sub> weight parameters |
| Whole-set Completeness | `completeness.test.unlearning_method.completeness_whole.final_value` | Max → 100 | Prediction agreement rate between M<sub>u</sub> and M<sub>r</sub> on D<sub>test</sub> |

#### Privacy Metrics

| Paper Metric | WandB Key Path | Optimal | Description |
|--------------|----------------|---------|-------------|
| MIA | `membership_inference_attack.test.unlearning_method.mia.final_value` | Closest to M<sub>r</sub> → 0 | Attack success rate identifying D<sub>f</sub> samples as former training data |
| ZRF | `zrf.test.unlearning_method.final_zrf.final_value` | Closest to M<sub>r</sub> → 1 | Similarity between M<sub>u</sub> and a randomly initialized teacher on D<sub>f</sub> |

#### Forgetting Metrics

| Paper Metric | WandB Key Path | Optimal | Description |
|--------------|----------------|---------|-------------|
| Forget-set Accuracy | `accuracy.test.unlearning_method.forget_acc.final_value` | Closest to M<sub>r</sub> → 0 | M<sub>u</sub> accuracy on D<sub>f</sub> |
| Forget-set Loss | `loss.test.unlearning_method.forget_loss.final_value` | Closest to M<sub>r</sub> → ∞ | M<sub>u</sub> cross-entropy loss on D<sub>f</sub> |
| Forget-set Completeness | `completeness.test.unlearning_method.completeness_forget.final_value` | Max → 100 | Prediction agreement between M<sub>u</sub> and M<sub>r</sub> on D<sub>f</sub> |

#### Utility Metrics

| Paper Metric | WandB Key Path | Optimal | Description |
|--------------|----------------|---------|-------------|
| Retain-set Accuracy | `accuracy.test.unlearning_method.retain_acc.final_value` | Closest to M<sub>r</sub> → 100 | M<sub>u</sub> accuracy on D<sub>r</sub> |
| Retain-set Loss | `loss.test.unlearning_method.retain_loss.final_value` | Closest to M<sub>r</sub> → 0 | M<sub>u</sub> cross-entropy loss on D<sub>r</sub> |
| Retain-set Completeness | `completeness.test.unlearning_method.completeness_retain.final_value` | Max → 100 | Prediction agreement between M<sub>u</sub> and M<sub>r</sub> on D<sub>r</sub> |
| Whole-set Accuracy | `accuracy.test.unlearning_method.whole_acc.final_value` | Closest to M<sub>r</sub> → 100 | M<sub>u</sub> accuracy on D<sub>test</sub> |
| Whole-set Loss | `loss.test.unlearning_method.whole_loss.final_value` | Closest to M<sub>r</sub> → 0 | M<sub>u</sub> cross-entropy loss on D<sub>test</sub> |

#### Efficiency Metrics

| Paper Metric | WandB Key Path | Optimal | Description |
|--------------|----------------|---------|-------------|
| Time (Speedup) | `time.unlearning_method.speedup.final_value` | Max (higher = faster) | Ratio T<sub>r</sub>/T<sub>u</sub> |
| Unlearning Time | `time.unlearning_method.core_time_elapsed.final_value` | Min → 0 | Raw unlearning execution time in seconds |

#### Resource Consumption Metrics

Resource metrics are embedded in all evaluation metrics under the `unlearning_method` category. Replace `<metric>` with any metric name.

| Paper Metric | WandB Key Path Pattern | Optimal | Description |
|--------------|------------------------|---------|-------------|
| CPU Memory Usage | `<metric>.*.unlearning_method.memory_usage.cpu_*` | Min → 0 | CPU RSS memory during unlearning (GB) |
| GPU Memory Usage | `<metric>.*.unlearning_method.memory_usage.gpu_*` | Min → 0 | GPU VRAM during unlearning (GB) |
| SM Utilisation | `<metric>.*.unlearning_method.compute_utilisation.*` | Min → 0 | GPU compute utilization (%) |

> `*` = `.test` for data-based metrics, nothing for parameter-based metrics.

---

## Per-Metric Field Paths

All fields store `{"final_value": <scalar>}` dictionaries (except `gpu_ids` which is a list).

---

## 1. Accuracy (`accuracy`)
Pattern: `accuracy.test.<category>.<field>`

Measures model performance by comparing predictions to ground truth labels.

**Metric value fields:**
```
accuracy.test.unlearning_method.whole_acc          # Accuracy on entire test set (should remain high)
accuracy.test.unlearning_method.retain_acc         # Accuracy on retain test set (should match Mo)
accuracy.test.unlearning_method.forget_acc         # Accuracy on forget test set (should approach 0)
```

**Unlearning resource fields (resources used BY the unlearning method):**
```
accuracy.test.unlearning_method.core_time_elapsed  # Time to perform unlearning (seconds)
accuracy.test.unlearning_method.memory_usage       # Peak memory during unlearning (GB)
accuracy.test.unlearning_method.compute_utilisation  # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields (resources used FOR computing accuracy):**
```
accuracy.test.eval_metric.core_time_elapsed        # Time to evaluate accuracy (seconds)
accuracy.test.eval_metric.memory_usage             # Peak memory for evaluation (GB)
accuracy.test.eval_metric.compute_utilisation        # GPU utilization for evaluation (%)
```

## 2. Loss (`loss`)
Pattern: `loss.test.<category>.<field>`

Measures cross-entropy loss indicating model uncertainty on predictions.

**Metric value fields:**
```
loss.test.unlearning_method.whole_loss             # CE loss on entire test set (should remain low)
loss.test.unlearning_method.retain_loss            # CE loss on retain set (should remain low)
loss.test.unlearning_method.forget_loss            # CE loss on forget set (should increase)
```

**Unlearning resource fields:**
```
loss.test.unlearning_method.core_time_elapsed      # Time to perform unlearning (seconds)
loss.test.unlearning_method.memory_usage           # Peak memory during unlearning (GB)
loss.test.unlearning_method.compute_utilisation      # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields:**
```
loss.test.eval_metric.core_time_elapsed            # Time to evaluate loss (seconds)
loss.test.eval_metric.memory_usage                 # Peak memory for evaluation (GB)
loss.test.eval_metric.compute_utilisation            # GPU utilization for evaluation (%)
```

## 3. ZRF (Zero Retrain Forgetting) (`zrf`)
Pattern: `zrf.test.<category>.<field>`

Measures similarity between model outputs and a randomly initialized model on forget set (1 - JS divergence).

**Metric value fields:**
```
zrf.test.unlearning_method.initial_zrf             # ZRF of Mo vs random teacher (baseline)
zrf.test.unlearning_method.final_zrf               # ZRF of Mu vs random teacher (should approach 1)
```

**Unlearning resource fields:**
```
zrf.test.unlearning_method.core_time_elapsed       # Time to perform unlearning (seconds)
zrf.test.unlearning_method.memory_usage            # Peak memory during unlearning (GB)
zrf.test.unlearning_method.compute_utilisation       # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields (nested for each ZRF computation):**
```
zrf.test.eval_metric.initial_zrf.core_time_elapsed # Time to compute initial ZRF (seconds)
zrf.test.eval_metric.initial_zrf.memory_usage      # Memory for initial ZRF computation (GB)
zrf.test.eval_metric.initial_zrf.compute_utilisation # GPU utilization for initial ZRF (%)
zrf.test.eval_metric.final_zrf.core_time_elapsed   # Time to compute final ZRF (seconds)
zrf.test.eval_metric.final_zrf.memory_usage        # Memory for final ZRF computation (GB)
zrf.test.eval_metric.final_zrf.compute_utilisation   # GPU utilization for final ZRF (%)
```

## 4. JS-Divergence (`jsdiv`)
Pattern: `jsdiv.test.<category>.<field>`

Measures Jensen-Shannon divergence between Mu and Mr output distributions on whole, retain, and forget sets.

**Metric value fields:**
```
jsdiv.test.unlearning_method.jsdiv_whole           # JS divergence on entire test set (lower is better, 0 = identical)
jsdiv.test.unlearning_method.jsdiv_retain          # JS divergence on retain test set
jsdiv.test.unlearning_method.jsdiv_forget          # JS divergence on forget test set
```

**Unlearning resource fields:**
```
jsdiv.test.unlearning_method.core_time_elapsed     # Time to perform unlearning (seconds)
jsdiv.test.unlearning_method.memory_usage          # Peak memory during unlearning (GB)
jsdiv.test.unlearning_method.compute_utilisation     # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields (nested per subset):**
```
jsdiv.test.eval_metric.jsdiv_whole.core_time_elapsed   # Time for whole-set JS divergence (seconds)
jsdiv.test.eval_metric.jsdiv_whole.memory_usage        # Memory for whole-set JS divergence (GB)
jsdiv.test.eval_metric.jsdiv_whole.compute_utilisation   # GPU utilization for whole-set JS divergence (%)
jsdiv.test.eval_metric.jsdiv_retain.core_time_elapsed  # Time for retain-set JS divergence (seconds)
jsdiv.test.eval_metric.jsdiv_retain.memory_usage       # Memory for retain-set JS divergence (GB)
jsdiv.test.eval_metric.jsdiv_retain.compute_utilisation  # GPU utilization for retain-set JS divergence (%)
jsdiv.test.eval_metric.jsdiv_forget.core_time_elapsed  # Time for forget-set JS divergence (seconds)
jsdiv.test.eval_metric.jsdiv_forget.memory_usage       # Memory for forget-set JS divergence (GB)
jsdiv.test.eval_metric.jsdiv_forget.compute_utilisation  # GPU utilization for forget-set JS divergence (%)
```

## 5. Membership Inference Attack (`membership_inference_attack`)
Pattern: `membership_inference_attack.test.<category>.<field>`

Measures privacy leakage by testing if forget samples can be identified as former training data.

**Metric value fields:**
```
membership_inference_attack.test.unlearning_method.mia                # Attack success rate (lower is better, should approach 0)
```

**Unlearning resource fields:**
```
membership_inference_attack.test.unlearning_method.core_time_elapsed  # Time to perform unlearning (seconds)
membership_inference_attack.test.unlearning_method.memory_usage       # Peak memory during unlearning (GB)
membership_inference_attack.test.unlearning_method.compute_utilisation  # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields:**
```
membership_inference_attack.test.eval_metric.core_time_elapsed        # Time to perform MIA (seconds)
membership_inference_attack.test.eval_metric.memory_usage             # Memory for MIA computation (GB)
membership_inference_attack.test.eval_metric.compute_utilisation        # GPU utilization for MIA (%)
```

## 6. Activation Distance (`activation_distance`)
Pattern: `activation_distance.test.<category>.<field>`

Measures L2 distance between Mu and Mr final-layer activations/outputs on whole, retain, and forget sets.

**Metric value fields:**
```
activation_distance.test.unlearning_method.activation_distance_whole     # L2 distance on entire test set (lower is better, 0 = identical)
activation_distance.test.unlearning_method.activation_distance_retain    # L2 distance on retain test set
activation_distance.test.unlearning_method.activation_distance_forget    # L2 distance on forget test set
```

**Unlearning resource fields:**
```
activation_distance.test.unlearning_method.core_time_elapsed             # Time to perform unlearning (seconds)
activation_distance.test.unlearning_method.memory_usage                  # Peak memory during unlearning (GB)
activation_distance.test.unlearning_method.compute_utilisation             # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields (nested per subset):**
```
activation_distance.test.eval_metric.activation_distance_whole.core_time_elapsed   # Time for whole-set computation (seconds)
activation_distance.test.eval_metric.activation_distance_whole.memory_usage        # Memory for whole-set computation (GB)
activation_distance.test.eval_metric.activation_distance_whole.compute_utilisation   # GPU utilization for whole-set computation (%)
activation_distance.test.eval_metric.activation_distance_retain.core_time_elapsed  # Time for retain-set computation (seconds)
activation_distance.test.eval_metric.activation_distance_retain.memory_usage       # Memory for retain-set computation (GB)
activation_distance.test.eval_metric.activation_distance_retain.compute_utilisation  # GPU utilization for retain-set computation (%)
activation_distance.test.eval_metric.activation_distance_forget.core_time_elapsed  # Time for forget-set computation (seconds)
activation_distance.test.eval_metric.activation_distance_forget.memory_usage       # Memory for forget-set computation (GB)
activation_distance.test.eval_metric.activation_distance_forget.compute_utilisation  # GPU utilization for forget-set computation (%)
```

## 7. Layerwise Distance (`layerwise_distance`)
Pattern: `layerwise_distance.<category>.<field>` (No `.test`)

Measures Euclidean distance between Mu and Mr model weights across all layers.

**Metric value fields:**
```
layerwise_distance.unlearning_method.layerwise_distance       # Weight distance (lower is better, 0 = identical weights)
```

**Unlearning resource fields:**
```
layerwise_distance.unlearning_method.core_time_elapsed        # Time to perform unlearning (seconds)
layerwise_distance.unlearning_method.memory_usage             # Peak memory during unlearning (GB)
layerwise_distance.unlearning_method.compute_utilisation        # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields:**
```
layerwise_distance.eval_metric.core_time_elapsed              # Time to compute weight distance (seconds)
layerwise_distance.eval_metric.memory_usage                   # Memory for distance computation (GB)
layerwise_distance.eval_metric.compute_utilisation              # GPU utilization for computation (%)
```

## 8. Completeness (`completeness`)
Pattern: `completeness.test.<category>.<field>`

Measures prediction consistency between Mu and Mr (agreement rate regardless of correctness).

**Metric value fields:**
```
completeness.test.unlearning_method.completeness_whole        # Agreement on entire test set (higher % is better)
completeness.test.unlearning_method.completeness_retain       # Agreement on retain test set (should be high)
completeness.test.unlearning_method.completeness_forget       # Agreement on forget test set (should be high)
```

**Unlearning resource fields:**
```
completeness.test.unlearning_method.core_time_elapsed         # Time to perform unlearning (seconds)
completeness.test.unlearning_method.memory_usage              # Peak memory during unlearning (GB)
completeness.test.unlearning_method.compute_utilisation         # GPU compute utilization during unlearning (%)
```

**Evaluation resource fields (nested for each subset):**
```
completeness.test.eval_metric.completeness_whole.core_time_elapsed   # Time for whole set evaluation (seconds)
completeness.test.eval_metric.completeness_whole.memory_usage        # Memory for whole set (GB)
completeness.test.eval_metric.completeness_whole.compute_utilisation   # GPU utilization for whole set (%)
completeness.test.eval_metric.completeness_retain.core_time_elapsed  # Time for retain set evaluation (seconds)
completeness.test.eval_metric.completeness_retain.memory_usage       # Memory for retain set (GB)
completeness.test.eval_metric.completeness_retain.compute_utilisation  # GPU utilization for retain set (%)
completeness.test.eval_metric.completeness_forget.core_time_elapsed  # Time for forget set evaluation (seconds)
completeness.test.eval_metric.completeness_forget.memory_usage       # Memory for forget set (GB)
completeness.test.eval_metric.completeness_forget.compute_utilisation  # GPU utilization for forget set (%)
```

## 9. Time (`time`)
Pattern: `time.<category>.<field>` (No `.test`)

Measures computational efficiency by comparing unlearning time to retraining time.

**Metric value fields:**
```
time.unlearning_method.core_time_elapsed                # Unlearning time Tu in seconds
time.unlearning_method.speedup                          # Speedup ratio Tr/Tu (higher is better, >1 means faster than retraining)
time.unlearning_method.gpu_ids                          # GPUs used during unlearning (list, for reproducibility)
```

**Unlearning resource fields:**
```
time.unlearning_method.memory_usage                     # Peak memory during unlearning (GB)
time.unlearning_method.compute_utilisation                # GPU compute utilization during unlearning (%)
```
**Note:** `core_time_elapsed` appears as both a metric value AND a resource field for time.

**Evaluation resource fields:**
```
time.eval_metric.core_time_elapsed                      # Time to compute speedup metric (seconds)
time.eval_metric.memory_usage                           # Memory for speedup computation (GB)
time.eval_metric.compute_utilisation                      # GPU utilization for computation (%)
```

## 10. Resource Consumption (Implementation Note)

**In Research Literature:** Resource Consumption is defined as a standalone evaluation metric (see paper Appendix Section on "Computational Efficiency Metrics") that directly measures memory usage and GPU compute utilization during the unlearning process.

**In This Codebase:** Rather than implementing it as a separate metric, resource consumption is **embedded** into all 9 metrics above through these common fields:
- `core_time_elapsed` - Execution time tracking (seconds)
- `memory_usage` - CPU RSS + GPU VRAM tracking (GB)
- `compute_utilisation` - GPU streaming multiprocessor utilization (%)

These fields appear in both:
- `unlearning_method` category: Resources consumed **BY** the unlearning method itself
- `eval_metric` category: Resources consumed **FOR** computing each evaluation metric

---

## Summary Notes

- **Field Structure**: All fields (except `gpu_ids`) store dictionaries with a `{"final_value": <scalar>}` structure plus potentially other metadata
- **Special Cases**:
  - **ZRF, JS-Divergence, Activation Distance, and Completeness**: Have nested resource tracking under `eval_metric` for their sub-measurements (e.g., `zrf.test.eval_metric.initial_zrf.core_time_elapsed`, `jsdiv.test.eval_metric.jsdiv_forget.core_time_elapsed`)
  - **Time**: The only metric with `gpu_ids` field (stores a list, not a dictionary with `final_value`)
  - **Time**: `core_time_elapsed` appears as both a metric value AND a resource field
- **Access Pattern**: To get the actual scalar value from WandB, access: `field_name["final_value"]` (except for `gpu_ids` which is already a list)
