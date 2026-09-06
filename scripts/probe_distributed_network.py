"""Bounded direct TCP reachability check before a multi-node GPU launch."""

import argparse
import json
import socket
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--host", default="::")
    parser.add_argument("--port", type=int, default=29683)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0 or not 0 <= args.port <= 65535:
        parser.error("Use a positive timeout and a valid TCP port")
    if args.server:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            listener.bind((args.host, args.port))
            listener.listen(4)
            deadline = time.monotonic() + args.timeout
            print(f"LISTENING {args.host}:{listener.getsockname()[1]}", flush=True)
            while time.monotonic() < deadline:
                listener.settimeout(max(0.01, deadline - time.monotonic()))
                try:
                    connection, address = listener.accept()
                except socket.timeout:
                    break
                with connection:
                    connection.settimeout(2)
                    connection.sendall(b"supreme-network-probe\n")
                print(f"CONNECTED {address}", flush=True)
    else:
        try:
            with socket.create_connection(
                (args.host, args.port), timeout=args.timeout
            ) as connection:
                with connection.makefile("rb") as stream:
                    reply = stream.readline(128)
                if reply != b"supreme-network-probe\n":
                    raise ValueError(f"Unexpected probe response: {reply!r}")
                print(
                    json.dumps(
                        {"host": args.host, "port": args.port, "connected": True}
                    )
                )
        except (OSError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "host": args.host,
                        "port": args.port,
                        "connected": False,
                        "error": str(exc),
                    }
                )
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
