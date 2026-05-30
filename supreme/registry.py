"""Central component registry for SUPREME.

SUPREME resolves datasets, models, unlearning methods/baselines and evaluation
metrics by **name**, using a convention: a registered name maps to a module at
``supreme.<subpackage>.<name>`` exposing a callable/class of the same name
(e.g. ``ResNet18`` -> ``supreme.models.ResNet18:ResNet18``). This module keeps
that convention intact while letting components be registered from **outside**
the package - with no edits to framework code - in two ways:

1. **Runtime API** (use in your own scripts before launching the pipeline)::

       import supreme
       supreme.register_unlearning_method("mymethod", "my_pkg.mymethod")
       supreme.register_model("MyNet", "my_pkg.models:MyNet")
       supreme.register_dataset("MyDS", "my_pkg.data:MyDS",
                                root="/data/myds", class_dict={"a": 0, "b": 1})

2. **Packaging entry points** (auto-discovered when a plugin package is
   installed). Direct ``module:attr`` groups for the callable categories::

       [project.entry-points."supreme.models"]
       MyNet = "my_pkg.models:MyNet"
       [project.entry-points."supreme.baselines"]
       mybase = "my_pkg.mybase:mybase"
       [project.entry-points."supreme.unlearning_methods"]
       mymethod = "my_pkg.mymethod:mymethod"
       [project.entry-points."supreme.metrics"]
       mymetric = "my_pkg.mymetric:mymetric"

   Datasets carry extra metadata (root dir, class dict, schedule), and bulk
   registration is sometimes convenient, so a ``supreme.plugins`` group points
   to a zero-argument setup callable that performs registration itself::

       [project.entry-points."supreme.plugins"]
       my_plugin = "my_pkg.register:setup"   # setup() calls supreme.register_*

Resolution order per category: **runtime overrides -> entry points -> built-in
convention**. Built-in components have no override and no entry point, so they
always fall through to the convention - their resolution is byte-for-byte
identical to the pre-packaging framework. This module imports only the standard
library and the (torch-free) ``project_config``, so ``import supreme`` and
registration work without importing the heavy PyTorch/Lightning stack.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata

from supreme.utils import project_config

# ---------------------------------------------------------------------------
# Convention configuration (single source of truth for the import prefix)
# ---------------------------------------------------------------------------

#: Top-level import package. Was hardcoded as the ``"src."`` / ``"supreme."``
#: prefix at each call site; centralised here so the convention has one home.
PACKAGE_ROOT = "supreme"

# category -> dotted sub-path under PACKAGE_ROOT used by the naming convention
_CONVENTION_SUBPATH = {
    "model": "models",
    "baseline": "methods.baselines",
    "unlearning_method": "methods.unlearning_methods",
    "metric": "eval_metrics",
}

# category -> packaging entry-point group for direct (module:attr) registration
_ENTRY_POINT_GROUPS = {
    "model": "supreme.models",
    "baseline": "supreme.baselines",
    "unlearning_method": "supreme.unlearning_methods",
    "metric": "supreme.metrics",
    "dataset": "supreme.datasets",
}

#: Entry-point group of zero-arg setup callables (datasets / bulk registration).
_PLUGIN_GROUP = "supreme.plugins"

# category -> project_config name-list attribute kept in sync on registration
_CONFIG_LIST = {
    "model": "model_names",
    "baseline": "baselines",
    "unlearning_method": "unlearning_methods",
    "metric": "evaluation_metrics",
    "dataset": "dataset_names",
}

#: Metric names that ``eval_metrics.metrics_main.get_metric_scores`` already
#: knows about - either dispatched by a built-in branch or intentionally a
#: no-op there (``resource_consumption`` is gathered via the
#: ``@track_evaluation_metric`` decorator, not a branch). Only names absent from
#: this set are routed through the external-metric fallback, so built-in metric
#: behaviour is never altered.
BUILTIN_METRICS = frozenset(
    {
        "accuracy",
        "loss",
        "zrf",
        "jsdiv",
        "membership_inference_attack",
        "activation_distance",
        "layerwise_distance",
        "completeness",
        "time",
        "resource_consumption",
    }
)

# ---------------------------------------------------------------------------
# Runtime override state (populated by register_* and by entry points)
# ---------------------------------------------------------------------------

# name -> target string ("module" or "module:attr")
_callable_overrides = {category: {} for category in _CONVENTION_SUBPATH}
# name -> {"target", "root", "class_dict"}; target may be a string or a class
_dataset_overrides = {}

_entry_points_loaded = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_target(target, default_attr):
    """Split a ``"module[:attr]"`` target into ``(module_name, attr_name)``.

    A bare module string defaults the attribute to ``default_attr`` (the
    registered name), matching the framework's filename==funcname==name rule.
    """
    target = target.strip()
    if ":" in target:
        module_name, attr = target.split(":", 1)
        return module_name.strip(), attr.strip()
    return target, default_attr


def _ensure_name_registered(category, name):
    """Append ``name`` to the relevant ``project_config`` list(s) if missing.

    Keeps argparse choices and ``name in project_config.<list>`` membership
    checks working for externally registered components without editing
    framework code.
    """
    names = getattr(project_config, _CONFIG_LIST[category])
    if name not in names:
        names.append(name)
    if category in ("baseline", "unlearning_method"):
        if name not in project_config.all_methods:
            project_config.all_methods.append(name)


def _entry_points_for(group):
    """Return entry points for ``group`` across importlib_metadata API versions."""
    try:  # Python >= 3.10
        return list(importlib_metadata.entry_points(group=group))
    except TypeError:  # Python 3.9 returns a dict keyed by group
        return list(importlib_metadata.entry_points().get(group, []))


def _load_entry_points():
    """Discover and register components advertised by installed plugin packages.

    Idempotent and lazy: runs once on first resolution. Runtime ``register_*``
    calls take precedence (entry points never overwrite an existing override).
    """
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    for category, group in _ENTRY_POINT_GROUPS.items():
        for ep in _entry_points_for(group):
            if category == "dataset":
                _dataset_overrides.setdefault(
                    ep.name, {"target": ep.value, "root": None, "class_dict": None}
                )
            else:
                _callable_overrides[category].setdefault(ep.name, ep.value)
            _ensure_name_registered(category, ep.name)

    # Setup-callable plugins register whatever they like via the runtime API.
    for ep in _entry_points_for(_PLUGIN_GROUP):
        ep.load()()


# ---------------------------------------------------------------------------
# Public registration API
# ---------------------------------------------------------------------------

def register_model(name, target):
    """Register a model factory. ``target``: ``"module"`` or ``"module:attr"``."""
    _callable_overrides["model"][name] = target
    _ensure_name_registered("model", name)


def register_baseline(name, target):
    """Register a baseline method. ``target``: ``"module"`` or ``"module:attr"``."""
    _callable_overrides["baseline"][name] = target
    _ensure_name_registered("baseline", name)


def register_unlearning_method(name, target):
    """Register an unlearning method. ``target``: ``"module"`` or ``"module:attr"``."""
    _callable_overrides["unlearning_method"][name] = target
    _ensure_name_registered("unlearning_method", name)


def register_metric(name, target, *, requires_retrain=False):
    """Register an evaluation metric.

    ``target``: ``"module"`` or ``"module:attr"`` pointing to a callable
    (typically decorated with ``@track_evaluation_metric``). Set
    ``requires_retrain=True`` if the metric needs the retrained reference model
    ``M_r`` (this adds it to ``project_config.metrics_requiring_retrain``, which
    triggers the retrain pipeline when the metric is requested).
    """
    _callable_overrides["metric"][name] = target
    _ensure_name_registered("metric", name)
    if requires_retrain:
        project_config.metrics_requiring_retrain.add(name)


def register_dataset(
    name,
    target,
    *,
    root=None,
    class_dict=None,
    rn_epochs=None,
    rn_milestones=None,
    vit_epochs=None,
    vit_milestones=None,
):
    """Register a dataset and its metadata.

    Args:
        name: Dataset name used on the command line / in configs.
        target: ``"module"`` / ``"module:ClassName"`` string, or a class object.
        root: Optional data root directory (overrides ``get_root_directory``).
        class_dict: Optional ``{class_name: int_label}`` mapping used by
            class/subclass unlearning. Stored on ``project_config`` as
            ``<name>_dict`` so existing lookups keep working.
        rn_epochs / rn_milestones / vit_epochs / vit_milestones: Optional
            per-architecture training schedule, stored on ``project_config`` as
            ``<name>_RN_EPOCHS`` etc. so ``train_main`` finds them.
    """
    _dataset_overrides[name] = {
        "target": target,
        "root": root,
        "class_dict": class_dict,
    }
    _ensure_name_registered("dataset", name)
    if class_dict is not None:
        setattr(project_config, f"{name}_dict", class_dict)
    schedule = {
        f"{name}_RN_EPOCHS": rn_epochs,
        f"{name}_RN_MILESTONES": rn_milestones,
        f"{name}_ViT_EPOCHS": vit_epochs,
        f"{name}_ViT_MILESTONES": vit_milestones,
    }
    for attr, value in schedule.items():
        if value is not None:
            setattr(project_config, attr, value)


# ---------------------------------------------------------------------------
# Resolution API (consumed by the framework's existing call sites)
# ---------------------------------------------------------------------------

def resolve_callable_location(category, name):
    """Return ``(module_name, attr_name)`` for a model/baseline/method/metric.

    Override -> entry point -> built-in convention. For built-ins this returns
    exactly ``("supreme.<subpath>.<name>", name)``, identical to the original
    hardcoded behaviour.
    """
    _load_entry_points()
    overrides = _callable_overrides[category]
    if name in overrides:
        return _split_target(overrides[name], default_attr=name)
    return f"{PACKAGE_ROOT}.{_CONVENTION_SUBPATH[category]}.{name}", name


def resolve_method_location(method_name):
    """Resolve a baseline or unlearning method name to ``(module_name, attr)``.

    Mirrors the original ``unlearn_main`` dispatch: baselines first, then
    unlearning methods; raises ``ValueError`` for unknown names.
    """
    _load_entry_points()
    if method_name in project_config.baselines:
        return resolve_callable_location("baseline", method_name)
    if method_name in project_config.unlearning_methods:
        return resolve_callable_location("unlearning_method", method_name)
    raise ValueError(f"Method {method_name} is not recognized.")


def resolve_metric_location(name):
    """Return ``(module_name, attr_name)`` for an evaluation metric."""
    return resolve_callable_location("metric", name)


def external_metric_names(eval_metrics):
    """Return requested metric names not handled by built-in dispatch branches."""
    return [m for m in eval_metrics if m not in BUILTIN_METRICS]


def resolve_dataset_class(name):
    """Return the dataset class for ``name``.

    For built-ins this is ``getattr(supreme.datasets.datasets, name)``,
    identical to the original lookup.
    """
    _load_entry_points()
    if name in _dataset_overrides:
        target = _dataset_overrides[name]["target"]
        if isinstance(target, str):
            module_name, attr = _split_target(target, default_attr=name)
            return getattr(importlib.import_module(module_name), attr)
        return target  # already a class object
    module = importlib.import_module(f"{PACKAGE_ROOT}.datasets.datasets")
    return getattr(module, name)


def resolve_dataset_root(name):
    """Return the override data root for ``name``, or ``None`` to use the default."""
    _load_entry_points()
    entry = _dataset_overrides.get(name)
    if entry and entry.get("root"):
        return entry["root"]
    return None


def get_external_dataset_dict_name(name):
    """Return the ``project_config`` attribute name holding ``name``'s class dict.

    ``None`` if ``name`` is a built-in (use ``project_config`` defaults) or has
    no registered class dict.
    """
    _load_entry_points()
    entry = _dataset_overrides.get(name)
    if entry and entry.get("class_dict") is not None:
        return f"{name}_dict"
    return None
