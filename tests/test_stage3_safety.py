"""Stage 3 setup, failure handling and historical-checkpoint regressions."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from supreme.eval_metrics.distributed import evaluation_valid_mask
from supreme.eval_metrics.layerwise_distance import element_interval, model_lay_dist
from supreme.utils.fabric.fabric_setup import setup_model_for_inference
from supreme.utils.generic_utils import create_dataloader
from supreme.utils.model_logging import _deep_merge
from supreme.utils.unlearning.evaluation_utils import (
    resource_log_fields,
    track_resources,
)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_zero_inference_has_no_optimizer_and_preserves_training_config(
    monkeypatch, stage
):
    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    config = {"zero_optimization": {"stage": stage}, "optimizer": {"type": "Adam"}}
    original = deepcopy(config)
    fabric = SimpleNamespace(
        strategy=SimpleNamespace(
            config=config, precision=SimpleNamespace(precision="bf16-true")
        ),
        device="cpu",
        broadcast=lambda value, src=0: value,
        print=lambda *a: None,
    )
    seen = []

    def setup(module):
        seen.append(deepcopy(fabric.strategy.config))
        return module

    fabric.setup_module = setup
    model = setup_model_for_inference(
        fabric, torch.nn.Linear(2, 2), f"deepspeed_stage{stage}"
    )
    assert not model.training
    if stage == 3:
        assert seen[0]["zero_optimization"]["stage"] == 3
        assert "optimizer" not in seen[0]
    else:
        assert not seen  # no dummy optimizer/gradient allocations
        assert model(torch.ones(1, 2)).dtype == torch.float32
        assert next(model.parameters()).dtype == torch.bfloat16
        assert not any(p.requires_grad for p in model.parameters())
    assert fabric.strategy.config is config and config == original


def test_zero_config_restored_after_setup_error(monkeypatch):
    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    config = {"zero_optimization": {"stage": 3}}
    fabric = SimpleNamespace(
        strategy=SimpleNamespace(config=config),
        setup_module=Mock(side_effect=ValueError("setup")),
    )
    with pytest.raises(ValueError, match="setup"):
        setup_model_for_inference(fabric, torch.nn.Linear(2, 2), "deepspeed_stage3")
    assert fabric.strategy.config is config


@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_evaluation_always_uses_forward_precision_wrapper(monkeypatch, strategy):
    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    fabric = SimpleNamespace(setup_module=Mock(side_effect=lambda model: model))
    model = torch.nn.Linear(2, 2)
    assert setup_model_for_inference(fabric, model, strategy) is model
    fabric.setup_module.assert_called_once_with(model)


def test_partial_resource_logs_do_not_invent_measurements():
    fields = resource_log_fields(
        {"total_gpu_memory": 1}, {"per_process": {"avg_cpu_util": [2]}}, True
    )
    assert fields == {"TotalGPUMemoryGB": 1, "PerProcessAverageCPUUtil": [2]}
    assert resource_log_fields(None, None) == {}


def test_resource_merge_preserves_gpu_and_cpu_per_rank(monkeypatch):
    import supreme.utils.unlearning.evaluation_utils as tracking

    fabric = SimpleNamespace(global_rank=0)
    for name in (
        "start_memory_tracking",
        "start_compute_util_tracking",
        "start_cpu_util_tracking",
    ):
        monkeypatch.setattr(tracking, name, lambda *a: None)
    monkeypatch.setattr(
        tracking,
        "memory_usage_in_gb",
        lambda *a, **k: (0, "result", {"per_process": [1]}),
    )
    monkeypatch.setattr(tracking, "track_memory_usage", lambda *a: {})
    monkeypatch.setattr(
        tracking,
        "track_compute_util_usage",
        lambda *a: {"per_process": {"avg_compute_util": [3]}},
    )
    monkeypatch.setattr(
        tracking,
        "track_cpu_util_usage",
        lambda *a: {"per_process": {"avg_cpu_util": [4]}},
    )
    assert track_resources(lambda: None, fabric)[3]["per_process"] == {
        "avg_compute_util": [3],
        "avg_cpu_util": [4],
    }


def test_reevaluation_replaces_stale_scores_but_keeps_other_metrics():
    result = {"accuracy": {"final_value": 10}, "other": 3}
    _deep_merge(result, {"accuracy": {"final_value": 90}})
    assert result == {"accuracy": {"final_value": 90}, "other": 3}


def test_error_handler_does_not_enter_collectives():
    from supreme.utils.debug_utils import handle_distributed_error

    fabric = SimpleNamespace(
        local_rank=1, global_rank=1, all_gather=Mock(), barrier=Mock()
    )
    with pytest.raises(RuntimeError, match="injected"):
        try:
            raise RuntimeError("injected")
        except RuntimeError as exc:
            handle_distributed_error(fabric, exc)
    fabric.all_gather.assert_not_called()
    fabric.barrier.assert_not_called()


def test_memory_monitor_exits_after_failure(monkeypatch):
    import supreme.utils.memory_utils as memory

    threads = []
    original = memory.Thread

    def track_thread(*args, **kwargs):
        thread = original(*args, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(memory, "Thread", track_thread)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fabric = SimpleNamespace(barrier=lambda: None)

    def fail(fabric):
        raise ValueError("injected")

    with pytest.raises(ValueError, match="injected"):
        memory.memory_usage_in_gb(fail, fabric)
    assert threads and all(not thread.is_alive() for thread in threads)


def test_dataloader_drop_last_is_rejected():
    dataset = TensorDataset(torch.arange(5))
    loader = DataLoader(dataset, batch_size=2, drop_last=True)
    with pytest.raises(RuntimeError, match="drop_last"):
        evaluation_valid_mask(SimpleNamespace(world_size=1), loader, 0, 2, "cpu")


def test_stage3_batch_size_is_per_device_only_in_evaluation(monkeypatch):
    dataset = TensorDataset(torch.arange(8))
    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    assert create_dataloader(dataset, 4, num_gpus=2, num_workers=0).batch_size == 4
    monkeypatch.setenv("PERFORM_EVALUATION", "false")
    assert create_dataloader(dataset, 4, num_gpus=2, num_workers=0).batch_size == 2


def test_layerwise_element_partitions_are_balanced_and_complete():
    intervals = [element_interval(11173962, rank, 4) for rank in range(4)]
    assert intervals[0][0] == 0 and intervals[-1][1] == 11173962
    assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    sizes = [end - start for start, end in intervals]
    assert max(sizes) - min(sizes) <= 1


def test_layerwise_rejects_mismatched_models_and_handles_alias():
    fabric = SimpleNamespace(device="cpu", global_rank=0, world_size=1)
    model = torch.nn.Linear(4, 2)
    result = model_lay_dist(fabric, model, model, do_global_aggregation=False)
    assert result["metric_value_dict"]["final_value"] == 0
    with pytest.raises(ValueError, match="shapes"):
        model_lay_dist(
            fabric, model, torch.nn.Linear(3, 2), do_global_aggregation=False
        )


@pytest.mark.parametrize(
    "precision,dtype",
    [
        ("bf16-true", torch.bfloat16),
        ("bf16-mixed", torch.bfloat16),
        ("32-true", torch.float32),
    ],
)
def test_fsdp_evaluation_uses_consistent_batchnorm_precision(
    monkeypatch, precision, dtype
):
    from supreme.utils.fabric.fabric_setup import _create_distributed_strategy

    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    strategy, _ = _create_distributed_strategy(
        {"distributed_strategy": "fsdp", "precision": precision}, False, "cuda"
    )
    assert strategy.mixed_precision.param_dtype == dtype
    assert strategy.mixed_precision.buffer_dtype == dtype
    assert strategy.mixed_precision._module_classes_to_ignore == ()


def test_failed_result_serialization_preserves_existing_json(monkeypatch, tmp_path):
    import json
    from supreme.utils.model_logging import save_evaluation_results

    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    fabric = SimpleNamespace(print=lambda *a: None)
    save_evaluation_results(fabric, "Finetune", {"accuracy": 90})
    path = tmp_path / "Finetune" / "Finetune_eval_results.json"
    with pytest.raises(TypeError):
        save_evaluation_results(fabric, "Finetune", {"bad_value": object()})
    assert json.loads(path.read_text()) == {"accuracy": 90}
    assert not list(path.parent.glob(".evaluation-*"))


@pytest.mark.parametrize(
    "metric", ["accuracy", "zrf", "activation", "completeness", "mia"]
)
def test_empty_evaluation_datasets_fail_explicitly(fake_fabric, metric):
    from supreme.eval_metrics.activation_distance import actv_dist
    from supreme.eval_metrics.completeness import calculate_completeness
    from supreme.eval_metrics.membership_inference_attack import (
        get_membership_attack_prob,
    )
    from supreme.eval_metrics.zrf import ZRF
    from supreme.utils.training.training_utils import evaluate

    labels = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(torch.empty(0, 2), labels, labels))
    model = torch.nn.Linear(2, 2)
    calls = {
        "accuracy": lambda: evaluate(
            fake_fabric, model, loader, do_global_aggregation=True
        ),
        "zrf": lambda: ZRF(fake_fabric, model, model, loader),
        "activation": lambda: actv_dist(fake_fabric, model, model, loader),
        "completeness": lambda: calculate_completeness(
            fake_fabric, model, model, loader
        ),
        "mia": lambda: get_membership_attack_prob(
            fake_fabric, 1, model, loader, loader, loader
        ),
    }
    with pytest.raises(ValueError):
        calls[metric]()


@pytest.mark.parametrize("strategy_name", ["ddp", "fsdp"])
def test_sampler_uses_launched_world_not_visible_device_count(
    monkeypatch, strategy_name
):
    from supreme.utils.fabric.fabric_setup import _create_distributed_strategy

    monkeypatch.setenv("PERFORM_EVALUATION", "true")
    strategy, _ = _create_distributed_strategy(
        {"distributed_strategy": strategy_name, "precision": "32-true"}, False, "cuda"
    )
    strategy.parallel_devices = [torch.device("cuda:0")]
    strategy.cluster_environment = SimpleNamespace(
        world_size=lambda: 4, global_rank=lambda: 3
    )
    assert strategy.distributed_sampler_kwargs == {"num_replicas": 4, "rank": 3}
