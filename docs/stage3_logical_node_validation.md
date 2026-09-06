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

### Native transport

| Strategy | Numerical precision/batch cases | Expected failure cases |
| --- | ---: | ---: |
| DDP | 10 passed | 2 passed |
| FSDP | 10 passed | 2 passed |
| ZeRO 1 | 10 passed | 2 passed |
| ZeRO 2 | 10 passed | 2 passed |
| ZeRO 3 | 10 passed | 2 passed |

Each strategy covered all five precision modes crossed with both batch
policies. Every normal case had two successful agent exits and all four
expected rank completion and layout records. Global batch 8 resolved to 2
samples per GPU; per-device batch 4 resolved to global microbatch 16. The two
expected failures per strategy were a worker failure on the second logical
node and a classifier-fitting failure on global rank zero. Both were detected
and the test job exited without reaching its deadline.

DDP and FSDP results were retrieved and independently revalidated from the
earlier run. After access was restored, all three ZeRO stages were rerun in a
complete 36-case matrix. Its summary, agent exit codes, rank records and failure
markers were independently rechecked. These totals combine completed cases
across runs; the original interrupted 60-case run is not presented as a clean,
completed matrix.

### Socket-only transport

All five strategies passed the `bf16-true`, per-device-batch numerical probe
and both injected-failure checks: five normal cases plus ten expected failures
in a completed 15-case matrix. Its summary, exit codes, rank records and
failure markers were independently revalidated.

For every normal case, both agents' NCCL logs contained collective-channel
connections through `NET/Socket`, with no channel connections through P2P or
shared memory. This confirms actual socket transport, not merely a socket
used for rendezvous. It remains a loopback test on one physical host.

Combined accepted coverage is 55 numerical cases and 20 expected-failure cases
across native and socket-only transport. Repeated smoke tests and the partial
ZeRO 1 results from the interrupted run are not added to these totals.

### Supporting checks and retained failures

- The final local and pinned GPU-host CPU suites each passed 203 tests,
  including the terminal-signal cleanup regression. The updated handler was
  deployed before the successful ZeRO rerun. No real terminal hangup was
  injected into that GPU matrix.
- Checksums of all 76 tracked Python/shell source files under `src` matched
  between the test checkout and the publication-safe branch.
- Formatting and Markdown privacy checks passed. Raw logs remain private.
- An initial three-case DDP smoke test passed. A subsequent expanded run's first
  GPU job exited successfully, but adjacent rank log messages exposed a JSON
  parser bug in the test supervisor. The parser and its regression test were
  corrected before the accepted results above.
- Monitoring access was lost during the original run. Recovered logs confirmed
  nine successful ZeRO 1 numerical cases, but that run had not reached ZeRO 2
  or ZeRO 3. No test workers remained when access returned. The interruption
  does not establish a GPU collective failure; the subsequent ZeRO matrix
  completed successfully.

Final process checks after the socket-only matrix found no surviving logical-
node supervisors, torchrun agents or probe workers. The four selected GPUs
returned to their initial free-memory levels. No server configuration changes
or actions against other users' processes were required. The physical
multi-host validation boundary described above is unchanged.
