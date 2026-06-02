"""Registry resolution contract tests (pure stdlib, no torch).

The registry is the framework's spine: every component is found by name through
``runtime override -> entry point -> built-in convention``. These tests pin that
contract - that every built-in name resolves to a real callable/class, that a
runtime registration wins over the convention, and that unknown names fail
loudly - without importing the heavy stack.
"""

import importlib

import pytest

from supreme import registry
from supreme.utils import project_config


def _load(module_name, attr):
    return getattr(importlib.import_module(module_name), attr)


# Module-level callables for the live-registration tests. A registered live
# callable resolves via "<__module__>:<__name__>", so it must be reachable as a
# real module attribute (a function nested inside a test is not).
def gentle_finetune():
    pass


def my_metric():
    pass


# --- built-in components resolve to importable callables -------------------


@pytest.mark.parametrize("name", project_config.model_names)
def test_builtin_models_resolve(name):
    module_name, attr = registry.resolve_callable_location("model", name)
    assert callable(_load(module_name, attr))


@pytest.mark.parametrize("name", project_config.all_methods)
def test_builtin_methods_resolve(name):
    # Covers baselines and unlearning methods via the same dispatch unlearn_main
    # uses (baselines first, then unlearning methods).
    module_name, attr = registry.resolve_method_location(name)
    assert callable(_load(module_name, attr))


def test_builtin_metrics_are_dispatched_not_routed_externally():
    # Built-in metrics are handled by metrics_main's own branches, so none of
    # them should be flagged as "external". Their in-file function names don't
    # follow the filename==funcname convention (e.g. activation_distance ->
    # actv_dist), which is exactly why they must not be resolved by convention.
    assert registry.external_metric_names(project_config.evaluation_metrics) == []


# --- runtime overrides win over the convention -----------------------------


def test_runtime_override_takes_precedence():
    registry.register_model("ResNet18", "my_pkg.models:CustomNet")
    assert registry.resolve_callable_location("model", "ResNet18") == (
        "my_pkg.models",
        "CustomNet",
    )


def test_bare_module_target_defaults_attr_to_name():
    registry.register_unlearning_method("mymethod", "my_pkg.mymethod")
    assert registry.resolve_callable_location("unlearning_method", "mymethod") == (
        "my_pkg.mymethod",
        "mymethod",
    )


def test_live_callable_registers_as_module_colon_name():
    registry.register_unlearning_method("gentle", gentle_finetune)
    module_name, attr = registry.resolve_method_location("gentle")
    assert _load(module_name, attr) is gentle_finetune


def test_registering_a_method_updates_config_lists():
    registry.register_unlearning_method("brand_new", "my_pkg.brand_new")
    assert "brand_new" in project_config.unlearning_methods
    assert "brand_new" in project_config.all_methods


# --- external metrics ------------------------------------------------------


def test_external_metric_is_listed_and_resolves():
    registry.register_metric("my_metric", my_metric)
    assert registry.external_metric_names(["accuracy", "my_metric"]) == ["my_metric"]
    module_name, attr = registry.resolve_metric_location("my_metric")
    assert _load(module_name, attr) is my_metric


def test_metric_requiring_retrain_is_tracked():
    registry.register_metric("needs_mr", "my_pkg.needs_mr", requires_retrain=True)
    assert "needs_mr" in project_config.metrics_requiring_retrain


# --- datasets --------------------------------------------------------------


@pytest.mark.parametrize("name", project_config.dataset_names)
def test_builtin_dataset_classes_resolve(name):
    cls = registry.resolve_dataset_class(name)
    assert isinstance(cls, type)


def test_registered_dataset_class_object_resolves_directly():
    class MyDataset:
        pass

    registry.register_dataset("MyDS", MyDataset, root="/data/myds")
    assert registry.resolve_dataset_class("MyDS") is MyDataset
    assert registry.resolve_dataset_root("MyDS") == "/data/myds"


# --- error handling --------------------------------------------------------


def test_unknown_method_raises_valueerror():
    with pytest.raises(ValueError):
        registry.resolve_method_location("this_method_does_not_exist")
