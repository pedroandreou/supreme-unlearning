# Stage 3 logical-node validation

## Scope and isolation

Two independent torchrun agents can represent two logical nodes on one physical
GPU host. Each agent sees a disjoint set of GPUs and launches one process per
visible device. Both agents join the same distributed process group. This
exercises the external multi-agent launch path without changing the host's
physical or administrative structure.

The test runner only creates ordinary user processes, temporary logs and
process-local environment variables. It does not create containers, VMs,
network namespaces or GPU partitions. It does not change firewall rules,
network interfaces, driver settings, system packages, GPU compute modes or
other users' processes. GPU visibility is a process-level selection, not an
exclusive reservation: use only devices assigned or approved for the test.
Small probes still consume GPU time, memory, CPU and local communication
bandwidth. These checks are not zero-impact or controlled speed benchmarks.

## Reproduction

Use the repository's pinned Python environment with PyTorch 2.1.0, Lightning
Fabric 2.1.0 and DeepSpeed 0.14.5. Select two equally sized, non-overlapping GPU
groups. The following indices are illustrative, not a resource allocation:

```bash
python scripts/validate_stage3_logical_nodes.py \
  --node0-gpus 0,1 --node1-gpus 2,3 \
  --precisions 32-true bf16-true bf16-mixed 16-true 16-mixed \
  --modes global per_device --failures
```

The runner starts two static-rendezvous torchrun agents with `nnodes=2`, node
ranks 0 and 1 and a shared loopback endpoint. It does not call the single-node
launcher twice to create two separate jobs. Each agent reports its workers'
global and local ranks, and the probe asserts that SUPREME infers two nodes
and the correct total world size.

A second diagnostic exercises local socket transport:

```bash
python scripts/validate_stage3_logical_nodes.py \
  --node0-gpus 0,1 --node1-gpus 2,3 \
  --precisions bf16-true --modes per_device --transport socket --failures
```

Socket mode sets `NCCL_NET=Socket`, `NCCL_P2P_DISABLE=1` and
`NCCL_SHM_DISABLE=1` only in the test subprocess environments. Communication
interfaces are restricted to loopback and InfiniBand is disabled for these
test subprocesses. Neither setting changes the server configuration or other
jobs. Inspect the private NCCL logs to confirm that collective channels use
the selected transport, not just that a bootstrap socket was opened.

## Acceptance checks

- DDP, FSDP and ZeRO 1/2/3 use one global process group across both agents.
- Each expected rank emits exactly one layout and completion record.
- Parameters are actually sharded for FSDP/ZeRO 3. DDP and optimizer-free
  ZeRO 1/2 evaluation retain full parameter replicas.
- Data-dependent numerical metrics match the unwrapped model oracle within
  explicit tolerances, and every real sample is counted once. One-sample and
  uneven 37-sample datasets exercise padded ranks and partial batches.
- MIA feature collection and the broadcast classifier score agree on all
  ranks. CPU classifier fitting remains centralized; it is not GPU-parallel.
- Empty datasets fail explicitly, and full-parameter distance calculations
  do not break later distributed inference.
- Global and per-device batch policies resolve against the full world size.
- Injected failures stop the whole test job, including the other agent, before
  the configured deadline. The outer test supervisor provides cross-agent
  cleanup; an independent production multi-node deployment still needs its
  scheduler or equivalent job-level supervision.

Raw logs and machine-specific evidence paths must remain private. The runner
stores them outside the repository in a temporary directory, with one log per
agent, per-case supervisor results, dependency versions and a final matrix
summary. Do not publish the raw logs without a separate sanitization review.

## Interpretation boundary

This test uses two logical nodes on one physical host, not two physical
servers. Even socket-only runs share a kernel, filesystem and loopback network.
They cannot validate inter-host routing, firewall access, independent storage,
network performance or physical-node loss. Successful results must not be
presented as physical multi-node or thousand-GPU scalability validation.

The earlier pinned-stack results and physical-network limitation are recorded
in [the pinned validation report](stage3_pinned_validation.md).

## Observed results, 6 September 2026

The experiment used two logical nodes of two GPUs each, four physical GPUs
on one host. Reported versions were Python 3.9.12, PyTorch 2.1.0+cu121,
Lightning Fabric 2.1.0 and DeepSpeed 0.14.5.

- DDP and FSDP each completed all ten precision/batch combinations and both
  injected-failure checks. Every normal case had two successful agent exits
  and all four expected rank completion and layout records.
- Seven ZeRO 1 numerical cases were observed passing before the monitoring
  connection stopped delivering output. New authenticated connections to both
  test hosts also timed out. This is not sufficient evidence to attribute the
  interruption to a GPU collective hang.
- The local and GPU-host CPU suites each passed 202 tests before the additional
  signal-handler regression test. The final local suite passed 203 tests after
  that addition. Formatting and Markdown privacy checks passed.
- An initial three-case DDP smoke test passed. A subsequent expanded run's first
  GPU job exited successfully, but adjacent rank log messages exposed a JSON
  parser bug in the test supervisor. The parser and its regression test were
  corrected before the matrix results above. That earlier matrix was not
  accepted as a completed run.

The full matrix, the planned socket-only run and final remote process cleanup
are **not yet verified**. Private on-host logs must be retrieved after access
is restored before these results can be marked complete. No replacement GPU
jobs were started after connectivity was lost. A terminal-hangup cleanup
handler was subsequently added locally; it has not yet been deployed or
GPU-tested on the remote host.
