# Extending SUPREME

SUPREME is **registry-based**: datasets, models, unlearning methods/baselines
and evaluation metrics are resolved by *name* through a convention - a
registered name `Foo` maps to `supreme.<subpackage>.Foo` exposing a
callable/class named `Foo`. You extend the framework by *implementing the
relevant interface and registering a module path* - never by editing framework
internals.

There are **two ways to register**, both requiring no edits to SUPREME's code:

- **Option A - From your own package (recommended for reuse).** Register from
  outside the installed `supreme` package, via the runtime API or packaging
  entry points. Covered in [Registering from outside the package](#registering-from-outside-the-package).
- **Option B - In-tree.** Add files under `supreme/` and list the name in
  [`supreme/utils/project_config.py`](../supreme/utils/project_config.py). Useful
  when you are vendoring/forking the framework. Covered in the per-component
  walkthroughs below.

Both paths share the same component interfaces (the contract your dataset
class / model factory / method function / metric function must satisfy). The
walkthroughs document those interfaces in detail; Option A simply changes
*where the files live* and *how the name is registered*.

---

## Registering from outside the package

> **Runnable notebook.** [`notebooks/custom_components.ipynb`](../notebooks/custom_components.ipynb)
> demonstrates this whole flow - installing SUPREME via pip and registering your
> own components - with a GPU-free proof cell you can run immediately.

Install SUPREME (`pip install supreme`) and your own package alongside it, then
register your components. Resolution order is **runtime overrides -> entry
points -> built-in convention**, so your registrations never collide with or
alter built-in components.

### Runtime API

Call these before launching the pipeline (e.g. at the top of your run script).
`target` is either a bare module (`"my_pkg.mymethod"`, attribute defaults to the
registered name) or an explicit `"module:attribute"`:

```python
import supreme

supreme.register_model("MyNet", "my_pkg.models:MyNet")
supreme.register_baseline("mybase", "my_pkg.mybase")
supreme.register_unlearning_method("mymethod", "my_pkg.mymethod")
supreme.register_metric("mymetric", "my_pkg.mymetric", requires_retrain=False)
supreme.register_dataset(
    "MyDS",
    "my_pkg.data:MyDS",
    root="/data/myds",                  # optional data root (else default layout)
    class_dict={"cat": 0, "dog": 1},    # for full/sub-class unlearning strategies
    rn_epochs=100, rn_milestones=[30, 60, 80],   # ResNet schedule (optional)
    vit_epochs=8,  vit_milestones=[7],            # ViT schedule (optional)
)

supreme.run_unlearning(["-method", "mymethod", "-net", "MyNet",
                        "-dataset", "MyDS", "-seed", "260"])
```

Registration keeps the framework's bookkeeping in sync automatically: the name
is appended to the relevant `project_config` list (so argument parsing and
validation accept it), a registered dataset's `class_dict` and training
schedule are attached to `project_config`, and a metric registered with
`requires_retrain=True` is added to `metrics_requiring_retrain` (which triggers
the retrained-reference `M_r` pipeline when the metric is requested).

### Packaging entry points (auto-discovered plugins)

A separately installed package can advertise components declaratively; SUPREME
discovers them on first use - no run-script changes needed. Use the direct
`module:attribute` groups for the callable categories:

```toml
# in your plugin package's pyproject.toml
[project.entry-points."supreme.models"]
MyNet = "my_pkg.models:MyNet"

[project.entry-points."supreme.baselines"]
mybase = "my_pkg.mybase:mybase"

[project.entry-points."supreme.unlearning_methods"]
mymethod = "my_pkg.mymethod:mymethod"

[project.entry-points."supreme.metrics"]
mymetric = "my_pkg.mymetric:mymetric"
```

Datasets carry extra metadata (root, class dict, schedule) that doesn't fit a
single entry-point value, and you may want to register several components at
once. For those, point the `supreme.plugins` group at a zero-argument **setup
callable** that performs registration via the runtime API:

```toml
[project.entry-points."supreme.plugins"]
my_plugin = "my_pkg.register:setup"
```

```python
# my_pkg/register.py
import supreme

def setup():
    supreme.register_dataset("MyDS", "my_pkg.data:MyDS",
                             root="/data/myds", class_dict={"cat": 0, "dog": 1})
    supreme.register_unlearning_method("mymethod", "my_pkg.mymethod")
```

> **Externally registered metrics.** Built-in metrics are dispatched by
> `supreme/eval_metrics/metrics_main.py`. Any requested metric name that is not
> built in is resolved through the registry and invoked automatically, so your
> metric runs with no edits to that file. Decorate it with
> `@track_evaluation_metric` (as the built-ins do) so it returns the standard
> result envelope and gets memory/power/time tracking; its result is recorded
> under its registered name.

The component interfaces themselves (signatures, Fabric-integration rules,
distributed-synchronisation requirements) are identical to the in-tree path and
are documented in full below.

---

- [Adding a new dataset](#adding-a-new-dataset)
- [Adding a new model](#adding-a-new-model)
- [Adding a new unlearning method](#adding-a-new-unlearning-method)
- [Adding a new evaluation metric](#adding-a-new-evaluation-metric)

---

## Adding a new dataset

### 1. Register the dataset name

In [`supreme/utils/project_config.py`](../supreme/utils/project_config.py):

```python
dataset_names = [
    "Cifar10",
    "Cifar20",
    "Cifar100",
    "PinsFaceRecognition",
    "Caltech101",
    "YourNewDataset",  # add here
]
```

### 2. Place dataset files and register the directory

Put your files under `supreme/datasets/data/your_dataset_directory/`, then map the dataset name to the directory in [`supreme/utils/generic_utils.py`](../supreme/utils/generic_utils.py):

```python
def get_root_directory(dataset_name):
    if dataset_name == "YourNewDataset":
        return "supreme/datasets/data/your_dataset_directory/"
    # ... existing mappings
```

### 3. Define the class dictionary

In [`supreme/utils/project_config.py`](../supreme/utils/project_config.py), add a dict mapping class names to integer labels (search for `cifar20_dict` for the pattern):

```python
your_new_dataset_dict = {
    "class_name_1": 0,
    "class_name_2": 1,
    # ... every class you want to support for unlearning
}
```

Then register it in `get_dict_name_for_dataset()`:

```python
dataset_to_dict = {
    "Cifar20": "cifar20_dict",
    "Cifar100": "cifar100_dict",
    "PinsFaceRecognition": "pins_dict",
    "Caltech101": "caltech101_dict",
    "YourNewDataset": "your_new_dataset_dict",  # add here
}
```

### 4. Compute and add normalisation constants

Each dataset needs its own per-channel mean/std. Add your dataset to [`supreme/utils/compute_dataset_stats.py`](../supreme/utils/compute_dataset_stats.py), then run:

```bash
python supreme/utils/compute_dataset_stats.py
```

Add the computed values at the top of [`supreme/datasets/datasets.py`](../supreme/datasets/datasets.py):

```python
YOUR_DATASET_MEAN = (...)
YOUR_DATASET_STD = (...)
```

Use them in the `Normalize` transform inside your dataset class. **Do not reuse another dataset's stats** - incorrect normalisation degrades model performance.

> **ResNet vs ViT normalisation:** ResNet18 trains from scratch and uses dataset-specific normalisation (the constants you computed). ViT loads pretrained ImageNet weights and uses ImageNet normalisation (`IMAGENET_MEAN` / `IMAGENET_STD`) regardless of the target dataset. The builder functions in `datasets.py` already follow this convention - make sure your class does too.

### 5. Implement the dataset class

In [`supreme/datasets/datasets.py`](../supreme/datasets/datasets.py):

```python
class YourNewDataset(Dataset):
    def __init__(self, ...):
        # Your implementation
        pass

    def __getitem__(self, idx):
        # Return data samples
        pass
```

Reference: see the `PinsFaceRecognition` class in [`supreme/datasets/datasets.py`](../supreme/datasets/datasets.py) for a complete example. For dataset-specific setup (e.g. manual downloads), see [`docs/adding_pinsfacerecognition.md`](adding_pinsfacerecognition.md).

---

## Adding a new model

### 1. Register the model name

In [`supreme/utils/project_config.py`](../supreme/utils/project_config.py):

```python
model_names = [
    "ResNet18",
    "ViT",
    "YourNewModel",  # add here
]
```

### 2. Create the model file

`supreme/models/YourNewModel.py`:

```python
def YourNewModel(args):
    # Return PyTorch model instance.
    # Must be compatible with Lightning Fabric (no special requirements).
    model = YourModelClass(...)
    return model
```

References: [`ResNet18.py`](../supreme/models/ResNet18.py) (built from scratch) and [`ViT.py`](../supreme/models/ViT.py) (loaded from HuggingFace).

---

## Adding a new unlearning method

> **AI-assistant warning.** AI assistants consistently misadapt PyTorch code to Lightning Fabric's distributed abstractions, even when given the Fabric docs. Use the steps below.

### 1. Register the method name

In [`supreme/utils/project_config.py`](../supreme/utils/project_config.py):

```python
unlearning_methods = [
    "retrain",
    "ssd",
    "random_labeling",
    "your_method",  # add here
]
```

The file name, the entry function name, and the registration name **must match**.

### 2. Create the method file

`supreme/methods/unlearning_methods/your_method.py`:

```python
def your_method(fabric, model, optimizer, retain_loader, forget_loader, **kwargs):
    # Your unlearning logic. Nothing needs to be returned.
    ...
```

### 3. Lightning Fabric integration

The 4 steps to scale a single-GPU implementation:

**Step 1 - import Fabric:**

```python
from lightning.fabric import Fabric
```

**Step 2 - set up models and optimisers.** Always pair a model with its optimiser; never set up the optimiser alone.

```python
# Single model, single optimiser (most common)
model, optimizer = fabric.setup(model, optimizer)

# Switching optimisers mid-process
raw_model = model.module if hasattr(model, "module") else model
model, new_optimizer = fabric.setup(raw_model, new_optimizer)

# Multiple models - set up separately
model1 = fabric.setup(model1)
model2 = fabric.setup(model2)
```

**FSDP deepcopy restriction.** Under FSDP, `fabric.setup()` replaces submodules with `FullyShardedDataParallel` wrappers containing NCCL process groups. `deepcopy()` on such a model fails with `TypeError: cannot pickle 'module' object`. Two safe alternatives:

```python
# (a) deepcopy BEFORE fabric.setup wraps the model
raw_model = model.module if hasattr(model, "module") else model
teacher_copy = deepcopy(raw_model)
model, optimizer = fabric.setup(raw_model, optimizer)
teacher = setup_model_for_inference(fabric, teacher_copy, distributed_strategy_name)

# (b) Create a fresh model and transfer weights via state_dict
from supreme.utils.fabric.fabric_setup import gather_full_state_dict, setup_model_for_inference
from supreme.utils.generic_utils import initialize_network
state = gather_full_state_dict(original_model)  # all ranks participate
teacher_raw = initialize_network(fabric=fabric, model_name=model_name, num_labels=num_labels, device=str(fabric.device))
teacher_raw.load_state_dict(state)
teacher = setup_model_for_inference(fabric, teacher_raw, distributed_strategy_name)
```

**Step 3 - set up DataLoaders.** Standard retain/forget train/test loaders are pre-configured by the framework (see [`supreme/utils/unlearning/unlearn_main.py`](../supreme/utils/unlearning/unlearn_main.py)). For custom loaders:

```python
loader1, loader2 = fabric.setup_dataloaders(loader1, loader2)
```

Examples: [UNSIR's custom dataloader](../supreme/methods/unlearning_methods/unsir.py), [Bad Teacher's custom dataloader](../supreme/methods/unlearning_methods/bad_teacher.py).

**Step 4 - code cleanup.** Remove all manual device management - Fabric handles it:

```python
# DELETE:
model.to(device)
batch.to(device)
tensor.cuda()
```

Use `fabric.backward(loss)` instead of `loss.backward()` for gradient synchronisation.

### 4. Distributed synchronisation for random operations

If your method uses any random operations, **synchronise them across processes** - otherwise each GPU generates different random values and the model diverges.

Operations to check:

| Library | Functions |
|---|---|
| PyTorch | `torch.rand`, `torch.randn`, `torch.randint`, `torch.randperm` |
| Python | `random.choice`, `random.sample`, `random.shuffle`, `random.randint` |
| NumPy | `np.random.*`, `np.percentile` (when computed independently per process) |

Fix pattern - generate on rank 0, broadcast to all:

```python
if fabric.global_rank == 0:
    random_tensor = torch.randn(batch_size, channels, height, width)
    random_indices = random.sample(range(len(dataset)), subset_size)
    indices_tensor = torch.tensor(random_indices, device=fabric.device)
else:
    random_tensor = torch.zeros(batch_size, channels, height, width, device=fabric.device)
    indices_tensor = torch.zeros(subset_size, dtype=torch.long, device=fabric.device)

random_tensor = fabric.broadcast(random_tensor, src=0)
indices_tensor = fabric.broadcast(indices_tensor, src=0)
random_indices = indices_tensor.cpu().tolist()
```

The same applies to computed thresholds (e.g., `np.percentile`) that can vary between processes due to floating-point ordering. Compute on rank 0 and broadcast.

Real examples in this codebase:
- [UNSIR noise synchronisation](../supreme/methods/unlearning_methods/unsir.py)
- [Bad Teacher subset sampling](../supreme/methods/unlearning_methods/bad_teacher.py)
- [Random Labeling label assignment](../supreme/methods/unlearning_methods/random_labeling.py)

### 5. Wire up method-specific arguments

In [`supreme/utils/unlearning/unlearn_main.py`](../supreme/utils/unlearning/unlearn_main.py), find the `elif method_name == "..."` block and add your method's custom kwargs:

```python
elif method_name == "your_method":
    kwargs.update({
        "retain_train_dataloader": retain_train_dataloader,
        "custom_temperature": 0.5,
        "alpha": 1.0,
        # any other parameters your method needs
    })
```

Then receive them in your method signature:

```python
def your_method(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,
    retain_train_dataloader: DataLoader,
    custom_temperature: float,
    alpha: float,
    **kwargs,  # catch any additional args
):
    ...
```

Base arguments (`fabric`, `wandb_logging_flag`, `type_of_unlearning_strategy`, `model`, `model_name`, `num_gpus`) are passed to every method automatically.

Example: see how [`bad_teacher`](../supreme/methods/unlearning_methods/bad_teacher.py) receives `unlearning_teacher`, `retain_train_dataloader`, and `forget_train_dataloader` on top of the base set.

### 6. Add the method to the execution list

In [`supreme/run_local.sh`](../supreme/run_local.sh) and [`supreme/run_slurm.sh`](../supreme/run_slurm.sh), append your method to `DEFAULT_METHODS`:

```bash
DEFAULT_METHODS="retrain,original,finetune,bad_teacher,random_labeling,unsir,ssd,lfssd,...,your_method"
```

### Reusable training loops

Save time with pre-built loops in [`supreme/utils/training/training_utils.py`](../supreme/utils/training/training_utils.py):

- `fit_one_learning_cycle` - standard fine-tuning. Used by [`finetune.py`](../supreme/methods/unlearning_methods/finetune.py), [`retrain.py`](../supreme/methods/baselines/retrain.py).
- `fit_one_unlearning_cycle` - unlearning iterations. Used by [`random_labeling.py`](../supreme/methods/unlearning_methods/random_labeling.py), [`unsir.py`](../supreme/methods/unlearning_methods/unsir.py).

Reference: [Lightning Fabric - Converting PyTorch Code](https://lightning.ai/docs/fabric/stable/fundamentals/convert.html).

---

## Adding a new evaluation metric

### 1. Register the metric name

In [`supreme/utils/project_config.py`](../supreme/utils/project_config.py):

```python
evaluation_metrics = [
    "accuracy",
    "activation_distance",
    # ...
    "your_metric",  # add here
]

# If your metric needs a retrained reference (M_r), also add it here.
# This is the set that triggers the retrain pipeline when present in --eval_metrics.
metrics_requiring_retrain = {
    "activation_distance",
    # ...
    "your_metric",  # only if M_r is needed
}
```

### 2. Create the metric file

`supreme/eval_metrics/your_metric.py`:

```python
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric

@track_evaluation_metric  # enables automatic memory/power/time tracking
def your_metric(fabric, unlearned_model, test_loader, **kwargs):
    result = ...  # compute metric

    # Aggregate across distributed processes
    gathered_result = fabric.all_gather(result)
    final_result = gathered_result.mean()  # or .max() / .sum()
    return final_result
```

### 3. Import the metric

In [`supreme/eval_metrics/metrics_main.py`](../supreme/eval_metrics/metrics_main.py):

```python
from supreme.eval_metrics.your_metric import your_metric
```

### Distributed aggregation patterns

Results must be aggregated across processes using `fabric.all_gather()`:

- `.mean()` - average across processes (e.g., [accuracy.py](../supreme/eval_metrics/accuracy.py), [zrf.py](../supreme/eval_metrics/zrf.py))
- `.sum()` - sum across processes (e.g., total correct predictions)

**Model copying.** If your metric modifies model state, copy the unwrapped module: `deepcopy(unlearned_model.module if hasattr(unlearned_model, "module") else unlearned_model)`. The `hasattr` check is needed because FSDP/DeepSpeed inference-only models are moved to device without `fabric.setup()` wrapping (to avoid parameter-sharding issues), so they don't have a `.module` attribute. DDP models always have `.module`.

References:
- Metric without retrained reference: [`accuracy.py`](../supreme/eval_metrics/accuracy.py)
- Metric requiring retrained reference: [`activation_distance.py`](../supreme/eval_metrics/activation_distance.py)
- Entry point: [`metrics_main.py`](../supreme/eval_metrics/metrics_main.py)
