"""Only pre-worker port collisions may retry; worker failures must propagate."""

import pytest

from supreme.utils.fabric import launch_single_node as launcher


def test_static_agent_store_and_no_worker_restarts():
    commands = []
    launcher.launch(4, "worker.py", ["--strategy", "fsdp"], runner=commands.append)
    assert len(commands) == 1
    args = commands[0]
    assert "--rdzv-backend=static" in args
    assert "--max-restarts=0" in args
    assert "--nproc-per-node=4" in args
    assert args[-3:] == ["worker.py", "--strategy", "fsdp"]
    assert not any("standalone" in arg for arg in args)


def test_retry_only_an_agent_port_collision(monkeypatch):
    ports = iter([20001, 20002])
    monkeypatch.setattr(launcher, "available_local_port", lambda: next(ports))
    calls = []

    def run(args):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("Address already in use")

    with pytest.warns(UserWarning, match="before startup"):
        launcher.launch(2, "worker.py", [], runner=run)
    assert len(calls) == 2
    assert "--master-port=20002" in calls[-1]


@pytest.mark.parametrize(
    "error", [RuntimeError("CUDA out of memory"), Exception("worker failure")]
)
def test_never_retry_worker_or_unrelated_failures(error):
    calls = []

    def run(args):
        calls.append(args)
        raise error

    with pytest.raises(type(error), match=str(error)):
        launcher.launch(2, "worker.py", [], runner=run)
    assert len(calls) == 1


def test_collision_retries_are_bounded():
    calls = []

    def run(args):
        calls.append(args)
        raise RuntimeError("EADDRINUSE")

    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="EADDRINUSE"):
        launcher.launch(2, "worker.py", [], runner=run, attempts=2)
    assert len(calls) == 2


def test_invalid_process_count_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        launcher.launch(0, "worker.py", [])


def test_torchrun_worker_failure_is_not_a_retryable_runtime_error():
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

    assert not issubclass(ChildFailedError, RuntimeError)
