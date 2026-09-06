"""Two independent torchrun agents on one host, without host configuration edits.

This is a logical-node simulation, not a physical multi-host network test.
Raw logs stay in a private temporary directory and must not be published.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid


def device_groups(first, second):
    groups = [value.split(",") for value in (first, second)]
    flat = [item for group in groups for item in group]
    if any(not re.fullmatch(r"[0-9]+", item) for item in flat):
        raise ValueError("GPU groups must contain comma-separated numeric indices")
    groups = [[str(int(item)) for item in group] for group in groups]
    flat = [item for group in groups for item in group]
    if len(groups[0]) != len(groups[1]) or len(set(flat)) != len(flat):
        raise ValueError("GPU groups must have equal sizes and no repeated devices")
    return groups


def child_environment(group, node, transport):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SLURM_", "TORCHELASTIC_"))
        and key
        not in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        }
    }
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(group),
        STAGE3_LOGICAL_NODE=str(node),
        OMP_NUM_THREADS="1",
        DATALOADER_NUM_WORKERS="0",
        NCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
        NCCL_IB_DISABLE="1",
        NCCL_DEBUG="INFO",
    )
    if transport == "socket":
        env.update(NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1", NCCL_NET="Socket")
    return env


def stop_groups(processes):
    # Signal only the new sessions created by this runner, never another job.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
            time.sleep(0.1)
    for process in processes:
        process.wait()


def check_logs(logs, local_world, failure):
    content = "\n".join(logs)
    if failure:
        expected = (
            "Injected Stage 3 rank-local failure"
            if failure == "rank"
            else "Injected Stage 3 MIA classifier failure"
        )
        if expected not in content:
            raise RuntimeError("Expected injected failure missing from logs")
        return
    for node, log in enumerate(logs):
        expected_ranks = set(range(node * local_world, (node + 1) * local_world))
        passes = re.findall(r"STAGE3_PROBE_PASS rank=(\d+)", log)
        if len(passes) != local_world or set(map(int, passes)) != expected_ranks:
            raise RuntimeError("Missing or duplicate rank completion markers")
        reports = [
            json.loads(item) for item in re.findall(r"STAGE3_LAYOUT (\{[^\n]+?\})", log)
        ]
        if (
            len(reports) != local_world
            or {r["rank"] for r in reports} != expected_ranks
        ):
            raise RuntimeError("Missing or duplicate rank layout records")
        for report in reports:
            if (
                report["logical_node"] != node
                or report["inferred_nodes"] != 2
                or report["local_rank"] != report["rank"] - node * local_world
                or report["data_parallel_world_size"] != local_world * 2
            ):
                raise RuntimeError("Incorrect logical-node rank topology")


def run_case(args, groups, output, case, precision, mode, failure=None):
    name = "-".join([case, precision, mode, failure or "normal"])
    case_dir = output / name
    case_dir.mkdir()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    # Static agent store avoids a second worker-store port race in Torch 2.1.
    # The initial bind can still collide; such a failure is reported, not hidden.
    strategy_args = ["--strategy", case]
    if case.startswith("zero"):
        strategy_args = ["--strategy", "deepspeed", "--stage", case[-1]]
    local_world = len(groups[0])
    batch = local_world * 4 if mode == "global" else 4
    extra = ["--fail-rank", str(local_world * 2 - 1)] if failure == "rank" else []
    if failure == "mia":
        extra = ["--fail-mia-fit"]
    run_id = uuid.uuid4().hex
    processes = []
    timed_out = False
    started = time.monotonic()
    with ExitStack() as stack:
        try:
            for node, group in enumerate(groups):
                log = stack.enter_context((case_dir / f"node{node}.log").open("w"))
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--rdzv-backend=static",
                    "--nnodes=2",
                    f"--node-rank={node}",
                    f"--nproc-per-node={local_world}",
                    "--master-addr=127.0.0.1",
                    f"--master-port={port}",
                    f"--rdzv-id={run_id}",
                    "--max-restarts=0",
                    "scripts/probe_stage3_distributed.py",
                    *strategy_args,
                    "--precision",
                    precision,
                    "--samples",
                    "37",
                    "--batch-size",
                    str(batch),
                    "--batch-size-mode",
                    mode,
                    *extra,
                ]
                processes.append(
                    subprocess.Popen(
                        command,
                        env=child_environment(group, node, args.transport),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
            while any(p.poll() is None for p in processes):
                if any(p.poll() not in (None, 0) for p in processes):
                    break  # Job-level supervision must stop the other agent too.
                if time.monotonic() - started > args.timeout:
                    timed_out = True
                    break
                time.sleep(0.2)
            observed_codes = [p.poll() for p in processes]
        finally:
            stop_groups(processes)
    result = dict(
        case=name,
        transport=args.transport,
        elapsed_seconds=round(time.monotonic() - started, 3),
        observed_exit_codes=observed_codes,
        final_exit_codes=[p.returncode for p in processes],
        timed_out=timed_out,
    )
    (case_dir / "supervisor.json").write_text(json.dumps(result, indent=2) + "\n")
    if timed_out:
        raise RuntimeError(f"{name}: exceeded bounded deadline")
    failed = any(code not in (None, 0) for code in observed_codes)
    if bool(failure) != failed or (not failure and observed_codes != [0, 0]):
        raise RuntimeError(f"{name}: unexpected agent exit codes {observed_codes}")
    check_logs(
        [(case_dir / f"node{n}.log").read_text() for n in range(2)],
        local_world,
        failure,
    )
    print(json.dumps(dict(result, passed=True)), flush=True)
    return result


def install_signal_handlers():
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    # Preserve run_case's finally cleanup if a terminal/SSH hangup reaches us.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, terminate)


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node0-gpus", required=True)
    parser.add_argument("--node1-gpus", required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["ddp", "fsdp", "zero1", "zero2", "zero3"],
        default=["ddp", "fsdp", "zero1", "zero2", "zero3"],
    )
    parser.add_argument("--precisions", nargs="+", default=["32-true", "bf16-true"])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["global", "per_device"],
        default=["global", "per_device"],
    )
    parser.add_argument("--transport", choices=["native", "socket"], default="native")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    groups = device_groups(args.node0_gpus, args.node1_gpus)
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if "LOCAL_RANK" in os.environ or "SLURM_JOB_ID" in os.environ:
        parser.error(
            "Run directly within your assigned resources, not inside an existing launcher"
        )
    os.chdir(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = (
        str(Path.cwd() / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    output = Path(tempfile.mkdtemp(prefix="supreme-logical-nodes-"))
    print(f"Private evidence: {output}", flush=True)
    import torch
    import lightning
    import deepspeed

    (output / "versions.json").write_text(
        json.dumps(
            dict(
                python=sys.version.split()[0],
                torch=torch.__version__,
                fabric=lightning.__version__,
                deepspeed=deepspeed.__version__,
                logical_nodes=2,
                processes_per_node=len(groups[0]),
                physical_hosts=1,
                transport=args.transport,
            ),
            indent=2,
        )
        + "\n"
    )
    results = []
    for case in args.cases:
        for precision in args.precisions:
            for mode in args.modes:
                results.append(run_case(args, groups, output, case, precision, mode))
        if args.failures:
            for failure in ("rank", "mia"):
                results.append(
                    run_case(
                        args, groups, output, case, "bf16-true", "per_device", failure
                    )
                )
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"LOGICAL_NODE_MATRIX_PASS cases={len(results)}", flush=True)


if __name__ == "__main__":
    main()
