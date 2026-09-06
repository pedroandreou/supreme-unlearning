"""Regression checks for reproducible GPU-binding and network test tools."""

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("rank", range(8))
def test_bound_rank_preserves_global_rank_and_selects_one_gpu(monkeypatch, rank):
    spec = importlib.util.spec_from_file_location(
        "bound_probe", SCRIPTS / "probe_stage3_bound_rank.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The helper writes several environment variables, so restore the entire
    # mapping rather than allowing synthetic SLURM metadata to leak to tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ.update(
        LOCAL_RANK=str(rank % 4),
        LOCAL_WORLD_SIZE="4",
        WORLD_SIZE="8",
        RANK=str(rank),
        CUDA_VISIBLE_DEVICES="0,2,4,6",
    )
    module.configure_bound_rank()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(2 * (rank % 4))
    assert os.environ["SLURM_PROCID"] == str(rank)
    assert os.environ["SLURM_NNODES"] == "2"
    assert os.environ["SLURM_NODEID"] == str(rank // 4)
    assert os.environ["SLURM_NTASKS"] == "8"


def test_network_probe_detects_a_real_loopback_connection():
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as check:
            check.bind(("::1", 0))
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    command = [sys.executable, str(SCRIPTS / "probe_distributed_network.py")]
    server = subprocess.Popen(
        [*command, "--server", "--port", "0", "--timeout", "3"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = server.stdout.readline()
        assert line.startswith("LISTENING "), line
        port = line.strip().rsplit(":", 1)[1]
        client = subprocess.run(
            [*command, "--host", "127.0.0.1", "--port", port, "--timeout", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert client.returncode == 0, client.stderr
        assert json.loads(client.stdout)["connected"] is True
    finally:
        server.terminate()
        server.communicate(timeout=5)


def test_network_probe_rejects_unbounded_waits():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "probe_distributed_network.py"),
            "--timeout",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
