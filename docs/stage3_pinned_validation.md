# Stage 3 pinned-stack validation, 5–6 September 2026

This supplements the [distributed evaluation implementation guide](distributed_evaluation.md).
Tests used a development branch, existing checkpoints and isolated checkouts.
Neither the paper/slides nor the original checkpoints were changed. No training
was required. Infrastructure identifiers, internal addresses, account names and
private artifact locations are intentionally excluded from this public record.
Raw evidence is retained privately, not committed to the repository.

## Exact core dependency versions

| Environment | Python | PyTorch | Lightning Fabric | DeepSpeed |
| --- | --- | --- | --- | --- |
| Pinned CUDA checks | 3.9.12 | 2.1.0+cu121 | 2.1.0 | 0.14.5 |
| Additional compatibility checks | 3.10.19 | 2.4.1+cu121 | 2.1.0 | 0.15.0 |

The primary environment matches the repository's pins for all three core
distributed libraries. DeepSpeed 0.14.5 and missing test dependencies were
installed into a temporary overlay with `pip --target` and `DS_BUILD_OPS=0`.
This avoided replacing packages in the existing venv. Optional compiled,
NVMe and offload operators were not tested. Version/import-path records are
written by `scripts/validate_stage3_probes.sh` and retained privately.
Source imports were explicitly directed to the isolated test checkout,
avoiding the unrelated editable installation in the original server worktree.

## Existing-checkpoint comparisons

Each row includes one single-GPU baseline and DDP, FSDP, ZeRO 1, ZeRO 2 and
ZeRO 3 runs. All built-in evaluation metrics, including resource tracking,
completed. The numerical comparator checked the same 20 score keys, rejected
nonfinite values and excluded runtime/resource measurements.

| Model and scenario | Distributed GPUs | Precision | Batch per GPU | Maximum absolute score difference from baseline |
| --- | ---: | --- | ---: | ---: |
| ResNet18, CIFAR-10, random 1% forgetting | 2 | 32-true | 128 | 2.71625641202e-07 |
| ResNet18, CIFAR-10, random 1% forgetting | 4 | bf16-true | 128 | 2.38147176312e-07 |
| ViT, Caltech101, full-class elephant forgetting | 2 | bf16-true | 16 | 4.39567271471e-11 |

The comparison tolerance was `1e-5`, unchanged for these runs. This establishes
distribution equivalence within the stated tolerance for these checkpoints;
it is not a claim of bitwise equivalence across arbitrary batch shapes or of
scientific validity for every inherited metric definition.

ViT used the existing offline Hugging Face cache and existing model/dataset
artifacts. The test artifact tree contains symlinks to checkpoints and input
metadata, while evaluation JSON is written into new directories.

## Adversarial probes

The CUDA probe checks actual parameter layout, BatchNorm/precision behavior,
numerical aggregation, exact sample counts with padding, MIA feature collection,
identical-model distance and inference after temporarily gathering parameters.
It also checks empty datasets and supervised failures. All-gather participation
is tested on real NCCL process groups, not only mocks.

Completed checks include:

- The complete CPU suite on both the local machine and the GPU test host:
  185 passed on each. This includes real 2/3/4/8-process Gloo tests and launcher
  retry checks. This count precedes the logical-node tooling additions.
- A clean four-GPU matrix using the corrected launcher: all five strategies
  crossed with `32-true`, `bf16-true`, `bf16-mixed`, `16-true` and `16-mixed`
  passed (25 cases). All ten injected failures (one worker failure and one MIA
  classifier failure per strategy) exited nonzero without timeout. Normal
  cases rejected empty inputs and continued successfully afterwards.
- Two-GPU FP16 probes for all five strategies.
- Seven-GPU bfloat16 probes for all five strategies in both global and per-device
  batch modes. Global batch 14 resolved to 2 per GPU; per-device batch 4 resolved
  to nominal global batch 28. One-sample datasets exercised six ranks with no
  valid samples, while all seven ranks still participated in collectives. All
  five seven-GPU per-device cases also passed a fresh rerun with the corrected
  launcher after the startup change.
- Simulated one-visible-GPU-per-SLURM-task binding on two real GPUs, all five
  strategies, both batch modes, worker failures and MIA classifier failures.
  This exercises the custom environment and samplers, but is not a SLURM
  scheduler or multi-node execution test.
- The updated single-node launcher on the newer PyTorch 2.4.1/Fabric 2.1.0/
  DeepSpeed 0.15.0 environment, all five strategies on two GPUs.
- Direct `python unlearn_main.py` evaluation automatically launching two GPUs:
  all 20 scores matched the baseline within `1.12e-16`. A separate invalid-global-
  batch test exited nonzero promptly instead of hanging or silently resizing.

## Startup issue found during stress testing

A repeated launch failed before evaluation with `Address already in use` when
the PyTorch 2.1 worker attempted to bind its rendezvous port. In that version's
c10d/standalone path, the agent selects a second worker-store port and releases
the temporary socket before workers bind it. Static rendezvous instead lets
workers reuse the live agent-owned store. This behavior was checked against
the [PyTorch 2.1 agent source](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/distributed/elastic/agent/server/local_elastic_agent.py)
and [static rendezvous implementation](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py).

`supreme.utils.fabric.launch_single_node` now selects a local port and launches
torchrun with static rendezvous and `max_restarts=0`. A collision while the
parent agent initially binds can retry up to five times; worker failures are
not retried. Unit tests distinguish these cases. `MAIN.sh`, direct multi-GPU
evaluation startup and the single-node validation runners use this helper.
SLURM and externally managed multi-node launches are unchanged. `MAIN.sh` also
ensures its own checkout precedes unrelated installations on `PYTHONPATH`.

Example, using devices assigned to the job:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m supreme.utils.fabric.launch_single_node \
  --nproc-per-node=2 scripts/probe_stage3_distributed.py --strategy fsdp
```

## Failures retained in the private record

- An initial pinned ZeRO 3 checkpoint run failed with an NCCL socket error.
  A selected GPU's free memory subsequently became very low under another
  workload. That observation does not establish the error's cause. ZeRO 3
  subsequently passed a diagnostic retry and the complete clean matrix on
  devices with more headroom. The failed job was not counted as a pass.
- All 25 precision cases and ten injected-failure cases completed in an earlier
  matrix, but uploading an updated shell runner during execution broke its
  final parsing. The overall run was not accepted as a clean matrix.
- A subsequent launch-port collision led to the startup correction described
  above. It is not a proven explanation of the earlier NCCL socket error.

These were shared-server correctness tests, not exclusive-resource performance
benchmarks. The eighth GPU had insufficient free memory during the large-world
checks, so the safe physical test limit in that round was seven GPUs.

## Physical multi-node boundary

Authenticated access to both test hosts worked, but direct peer TCP probes
timed out in both directions. Fresh high-port checks reproduced the problem:
each host reached its own listener through its LAN address, but neither could
reach the peer listener over IPv4 or link-local IPv6. Probes were bounded and
shut down. These observations do not establish that every port is blocked or
identify a particular filtering rule.

No firewall, routing, privilege, network namespace or host security changes
were made. Actual physical multi-node validation still requires an approved
allocation with permitted peer rendezvous and NCCL/Gloo traffic. Opening just
the rendezvous endpoint does not establish peer collective connectivity; see
the [NVIDIA networking documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html).
Infrastructure-specific diagnostics and administrator requests are kept private.

Seven-GPU correctness, simulated GPU binding and 1,024-rank sampler arithmetic
do not establish thousand-GPU readiness. The dataset-broadcast, full-checkpoint/
full-parameter materialization and centralized MIA fitting limitations in the
implementation guide remain. Process checks at the end of the earlier round
found no surviving validation or network-probe workers.

## Logical-node follow-up

The [logical-node validation report](stage3_logical_node_validation.md) records
the separate two-agent experiment on one physical host. It exercises global
rank handling and distributed evaluation across logical nodes without changing
the server configuration. Its results must not be interpreted as physical
multi-host network validation.
