"""CPU checks for the process-local logical-node test supervisor."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location(
        "logical_nodes",
        Path(__file__).resolve().parents[1]
        / "scripts/validate_stage3_logical_nodes.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "first,second",
    [("0", "0"), ("0,1", "2"), ("", "1"), ("0,0", "1,2"), ("0", "00"), ("GPU-x", "1")],
)
def test_reject_unsafe_device_groups(runner, first, second):
    with pytest.raises(ValueError):
        runner.device_groups(first, second)


def test_disjoint_gpu_groups(runner):
    assert runner.device_groups("0,2", "4,6") == [["0", "2"], ["4", "6"]]


def test_terminal_signals_unwind_cleanup(runner, monkeypatch):
    handlers = {}
    monkeypatch.setattr(
        runner.signal, "signal", lambda sig, handler: handlers.update({sig: handler})
    )
    runner.install_signal_handlers()
    assert set(handlers) == {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    for sig, handler in handlers.items():
        with pytest.raises(SystemExit) as caught:
            handler(sig, None)
        assert caught.value.code == 128 + sig


def test_child_configuration_does_not_mutate_parent(runner, monkeypatch):
    monkeypatch.setenv("SLURM_NNODES", "16")
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    before = dict(os.environ)
    child = runner.child_environment(["2", "3"], 1, "socket")
    assert dict(os.environ) == before
    assert "SLURM_NNODES" not in child and "RANK" not in child
    assert child["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert child["STAGE3_LOGICAL_NODE"] == "1"
    assert child["NCCL_NET"] == "Socket"
    assert child["NCCL_P2P_DISABLE"] == child["NCCL_SHM_DISABLE"] == "1"
    assert child["NCCL_SOCKET_IFNAME"] == child["GLOO_SOCKET_IFNAME"] == "lo"


def logs_for_two_nodes():
    return [
        "\n".join(
            "STAGE3_LAYOUT "
            + json.dumps(
                dict(
                    rank=rank,
                    local_rank=rank % 2,
                    logical_node=node,
                    inferred_nodes=2,
                    data_parallel_world_size=4,
                )
            )
            + f"\nSTAGE3_PROBE_PASS rank={rank}"
            for rank in range(node * 2, node * 2 + 2)
        )
        for node in range(2)
    ]


def test_rank_topology_and_completions(runner):
    runner.check_logs(logs_for_two_nodes(), 2, None)


def test_adjacent_rank_output_does_not_merge_json(runner):
    runner.check_logs([log.replace("\n", "") for log in logs_for_two_nodes()], 2, None)


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "wrong_node", "wrong_world"]
)
def test_reject_bad_rank_evidence(runner, change):
    logs = logs_for_two_nodes()
    if change == "duplicate":
        logs[0] += "\nSTAGE3_PROBE_PASS rank=0"
    elif change == "missing":
        logs[1] = logs[1].replace("STAGE3_PROBE_PASS rank=3", "")
    elif change == "wrong_node":
        logs[1] = logs[1].replace('"logical_node": 1', '"logical_node": 0')
    else:
        logs[1] = logs[1].replace(
            '"data_parallel_world_size": 4', '"data_parallel_world_size": 2'
        )
    with pytest.raises(RuntimeError):
        runner.check_logs(logs, 2, None)


def test_failure_requires_injected_error(runner):
    with pytest.raises(RuntimeError):
        runner.check_logs(["unrelated startup failure", ""], 2, "rank")
    runner.check_logs(["Injected Stage 3 rank-local failure", ""], 2, "rank")


def test_cleanup_only_owns_its_new_sessions(runner):
    own = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        runner.stop_groups([own])
        assert own.poll() is not None
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=5)
        if own.poll() is None:
            own.kill()
            own.wait(timeout=5)
