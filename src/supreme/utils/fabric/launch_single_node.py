"""Supervised single-node launch using an agent-owned rendezvous store.

PyTorch 2.1's c10d/standalone launcher selects a second, temporarily unreserved
worker-store port. Static rendezvous lets workers reuse the agent's live store.
Only an agent startup bind collision is retried, never a failed worker/job.
Multi-node and SLURM launches remain the responsibility of their own launchers.
"""

import argparse
import socket
import uuid
import warnings


def available_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def launch(nproc, script, script_args, runner=None, attempts=5):
    if nproc < 1 or attempts < 1:
        raise ValueError("nproc and attempts must be positive")
    if runner is None:
        from torch.distributed.run import main as runner
    for attempt in range(attempts):
        port = available_local_port()
        command = [
            "--rdzv-backend=static",
            "--nnodes=1",
            "--node-rank=0",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            f"--rdzv-id={uuid.uuid4().hex}",
            f"--nproc-per-node={nproc}",
            "--max-restarts=0",
            script,
            *script_args,
        ]
        try:
            return runner(command)
        except RuntimeError as error:
            # Static rendezvous binds on the parent agent before it launches
            # workers. Worker failures use ChildFailedError, not RuntimeError.
            message = str(error).lower()
            port_collision = any(
                token in message
                for token in (
                    "address already in use",
                    "eaddrinuse",
                    "errno: 98",
                )
            )
            if not port_collision or attempt + 1 == attempts:
                raise
            warnings.warn(
                f"Rendezvous port {port} was taken before startup; selecting another port"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("script")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    launch(args.nproc_per_node, args.script, args.script_args)


if __name__ == "__main__":
    main()
