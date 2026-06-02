"""Shared pytest fixtures for the SUPREME test suite.

The suite is deliberately CPU-only and download-free so it runs in CI without a
GPU, dataset, or Hugging Face fetch. Two helpers live here:

* ``fake_fabric`` - a stand-in for a Lightning ``Fabric`` covering only the
  surface the eval metrics touch on a single process (``global_rank`` and an
  identity ``all_gather``). Enough to exercise the aggregation branches without
  spinning up a real distributed launcher.
* ``_isolate_registry`` (autouse) - snapshots the registry's module-level
  override state and the ``project_config`` name lists before each test and
  restores them after, so a test that calls ``supreme.register_*`` can't leak
  into the next one.
"""

import copy

import pytest

from supreme import registry
from supreme.utils import project_config


class FakeFabric:
    """Minimal single-process Fabric stub for metric tests.

    Only the attributes the eval metrics actually use are implemented. On one
    process ``all_gather`` is the identity, so aggregated values equal their
    local counterparts - which is exactly what the metric math expects.
    """

    global_rank = 0
    world_size = 1

    def all_gather(self, data):
        return data

    def print(self, *args, **kwargs):  # metrics occasionally log progress
        pass


@pytest.fixture
def fake_fabric():
    return FakeFabric()


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save/restore registry + project_config global state around every test.

    The registry keeps its overrides in module-level dicts and mirrors names
    into ``project_config`` lists/sets. Tests that register components mutate
    that shared state, so we deep-copy it before the test and put it back after.
    """
    callable_overrides = copy.deepcopy(registry._callable_overrides)
    dataset_overrides = copy.deepcopy(registry._dataset_overrides)
    entry_points_loaded = registry._entry_points_loaded

    config_lists = {
        name: list(getattr(project_config, name))
        for name in (
            "model_names",
            "baselines",
            "unlearning_methods",
            "all_methods",
            "dataset_names",
            "evaluation_metrics",
        )
    }
    metrics_requiring_retrain = set(project_config.metrics_requiring_retrain)

    yield

    registry._callable_overrides = callable_overrides
    registry._dataset_overrides = dataset_overrides
    registry._entry_points_loaded = entry_points_loaded
    for name, value in config_lists.items():
        getattr(project_config, name)[:] = value
    project_config.metrics_requiring_retrain.clear()
    project_config.metrics_requiring_retrain.update(metrics_requiring_retrain)
