"""Real-CUDA adversarial probe, launched with torchrun (no pytest required).

Exercises parameter sharding, BatchNorm, uneven and smaller-than-world datasets,
all built-in numerical metric paths, alias models, and rank-local failure exit.
Example: PERFORM_EVALUATION=true CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
scripts/probe_stage3_distributed.py --strategy fsdp --precision bf16-true
"""

import argparse
from copy import deepcopy
import json
import os

import torch
from torch.utils.data import TensorDataset

from supreme.eval_metrics.activation_distance import actv_dist
from supreme.eval_metrics.completeness import calculate_completeness
from supreme.eval_metrics.distributed import gather_rank_values
from supreme.eval_metrics.jsdiv import js_divergence_elements
from supreme.eval_metrics.layerwise_distance import model_lay_dist
from supreme.eval_metrics.membership_inference_attack import (
    entropy,
    get_membership_attack_data,
    get_membership_attack_prob,
)
from supreme.eval_metrics.zrf import ZRF
from supreme.utils.debug_utils import handle_distributed_error
from supreme.utils.fabric.fabric_setup import (
    get_slurm_node_count,
    initialize_fabric,
    setup_model_for_inference,
)
from supreme.utils.training.training_utils import evaluate
from supreme.utils.generic_utils import create_dataloader
from supreme.utils.batching import resolve_batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["ddp", "fsdp", "deepspeed"], required=True
    )
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--precision", default="bf16-true")
    parser.add_argument("--fail-rank", type=int)
    parser.add_argument("--fail-mia-fit", action="store_true")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--batch-size-mode", choices=["global", "per_device"], default="per_device"
    )
    args = parser.parse_args()
    os.environ["PERFORM_EVALUATION"] = "true"
    fabric, _, _, _, strategy = initialize_fabric(
        {
            "distributed_strategy": args.strategy,
            "deepspeed_stage": args.stage,
            "precision": args.precision,
            "num_gpus": torch.cuda.device_count(),
            "callbacks": [],
            "model_name": "stage3-probe",
        }
    )
    try:
        if "STAGE3_LOGICAL_NODE" in os.environ:
            logical_node = int(os.environ["STAGE3_LOGICAL_NODE"])
            local_world = int(os.environ["LOCAL_WORLD_SIZE"])
            assert get_slurm_node_count() == 2
            assert fabric.world_size == 2 * local_world
            assert fabric.global_rank == logical_node * local_world + fabric.local_rank
            assert torch.cuda.device_count() == local_world
        torch.manual_seed(123)
        dtype = {"bf16-true": torch.bfloat16, "16-true": torch.float16}.get(
            args.precision, torch.float32
        )
        if args.strategy == "deepspeed" and "mixed" in args.precision:
            # DeepSpeed inference stores low-precision weights even for its
            # mixed-precision selections; there is no fp32 optimizer master.
            dtype = torch.bfloat16 if "bf16" in args.precision else torch.float16
        first = (
            torch.nn.Sequential(
                torch.nn.Linear(512, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.Tanh(),
                torch.nn.Linear(512, 3),
            )
            .to(device=fabric.device, dtype=dtype)
            .eval()
        )
        second = deepcopy(first)
        with torch.no_grad():
            second[-1].bias[0] += 0.125
        expected_distance = (
            sum(
                (p.double() - q.double()).square().sum()
                for p, q in zip(first.parameters(), second.parameters())
            )
            .sqrt()
            .item()
        )
        logical_numel = sum(p.numel() for p in first.parameters())
        inputs = torch.randn(max(128, args.samples * 4), 512, device=fabric.device)
        # Baseline model outputs are calculated before any distributed wrapping.
        with (
            torch.no_grad(),
            torch.autocast(
                "cuda",
                enabled="mixed" in args.precision,
                dtype=torch.bfloat16 if "bf16" in args.precision else torch.float16,
            ),
        ):
            logits1, logits2 = (
                first(inputs.to(dtype)).float(),
                second(inputs.to(dtype)).float(),
            )
        # Exact classification assertions need a margin: low-precision GEMM
        # kernels can round nearly tied logits differently for different batch
        # shapes, even without distribution. Continuous scores still use the
        # unwrapped full-batch oracle with an explicit numerical tolerance.
        stable = torch.ones(len(inputs), dtype=torch.bool, device=fabric.device)
        for logits in (logits1, logits2):
            stable &= logits.topk(2, dim=1).values.diff(dim=1).abs().flatten() > 0.02
        assert stable.sum().item() >= args.samples
        inputs = inputs[stable][: args.samples]
        logits1, logits2 = (
            logits1[stable][: args.samples],
            logits2[stable][: args.samples],
        )
        first = setup_model_for_inference(fabric, first, strategy)
        second = setup_model_for_inference(fabric, second, strategy)
        parameters = list(first.parameters())
        if args.strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel

            assert any(isinstance(m, FullyShardedDataParallel) for m in first.modules())
            assert sum(p.numel() for p in parameters) < logical_numel
        elif args.strategy == "deepspeed":
            from deepspeed import DeepSpeedEngine

            if args.stage == 3:
                assert isinstance(first._forward_module, DeepSpeedEngine)
                assert all(hasattr(p, "ds_id") for p in parameters)
                assert any(p.ds_tensor.numel() < p.ds_numel for p in parameters)
            else:
                from supreme.utils.fabric.inference import ReplicatedEvaluationModule

                assert isinstance(first, ReplicatedEvaluationModule)
                assert not any(p.requires_grad for p in parameters)
                assert sum(p.numel() for p in parameters) == logical_numel
        report = {
            "rank": fabric.global_rank,
            "local_rank": fabric.local_rank,
            "logical_node": int(os.environ.get("STAGE3_LOGICAL_NODE", "-1")),
            "inferred_nodes": get_slurm_node_count(),
            "strategy": strategy,
            "precision": args.precision,
            **resolve_batch_size(
                args.batch_size, fabric.world_size, args.batch_size_mode
            ),
            "logical_parameters": logical_numel,
            "local_parameter_elements": sum(
                getattr(p, "ds_tensor", p).numel() for p in parameters
            ),
        }
        print("STAGE3_LAYOUT " + json.dumps(report), flush=True)

        if args.fail_mia_fit and fabric.global_rank == 0:
            from supreme.eval_metrics import membership_inference_attack as mia_module

            def fail_fit(*unused_args, **unused_kwargs):
                raise ValueError("Injected Stage 3 MIA classifier failure")

            mia_module.LogisticRegression.fit = fail_fit

        for size in (1, args.samples):
            labels = logits1[:size].argmax(1)
            dataset = TensorDataset(inputs[:size].cpu(), labels.cpu(), labels.cpu())
            loader = fabric.setup_dataloaders(
                create_dataloader(
                    dataset,
                    args.batch_size,
                    num_gpus=fabric.world_size,
                    num_workers=0,
                    is_training=False,
                    batch_size_mode=args.batch_size_mode,
                )
            )
            assert loader.batch_size == report["per_device_batch_size"]
            p, q = logits1[:size].softmax(1), logits2[:size].softmax(1)
            result = evaluate(fabric, first, loader, do_global_aggregation=True)[
                "metric_value_dict"
            ]
            assert result["Acc"]["final_value"] == 100, {
                "size": size,
                "accuracy": result["Acc"],
                "min_baseline_margin": logits1[:size]
                .topk(2, dim=1)
                .values.diff(dim=1)
                .abs()
                .min()
                .item(),
            }
            expected_loss = torch.nn.functional.cross_entropy(
                logits1[:size], labels
            ).item()
            assert abs(result["Loss"]["final_value"] - expected_loss) < 0.004
            zrf = ZRF(fabric, first, second, loader)["metric_value_dict"]["final_value"]
            assert abs(zrf - (1 - js_divergence_elements(p, q).mean().item())) < 0.004
            activation = actv_dist(fabric, first, second, loader)["metric_value_dict"][
                "final_value"
            ]
            assert (
                abs(activation - (p - q).square().sum(1).mean().sqrt().item()) < 0.004
            )
            complete = calculate_completeness(fabric, first, second, loader)[
                "metric_value_dict"
            ]
            assert sum(complete["per_process_total_samples"]) == size
            assert (
                abs(
                    complete["final_value"]
                    - (logits1[:size].argmax(1) == logits2[:size].argmax(1))
                    .double()
                    .mean()
                    .item()
                    * 100
                )
                < 1e-6
            )
            features, _, _ = get_membership_attack_data(
                fabric, fabric.world_size, loader, loader, loader, first, True
            )
            torch.testing.assert_close(
                features.flatten().sort().values,
                entropy(p).sort().values,
                atol=0.004,
                rtol=0,
            )
            mia = get_membership_attack_prob(
                fabric, fabric.world_size, first, loader, loader, loader
            )["metric_value_dict"]["final_value"]
            assert 0 <= mia <= 1
            all_mia = gather_rank_values(
                fabric, torch.tensor([mia], device=fabric.device)
            )
            assert torch.all(all_mia == all_mia[0])

        # Empty inputs must fail explicitly on all ranks, not return an invalid
        # score or leave peers waiting in a mismatched collective.
        empty_labels = torch.empty(0, dtype=torch.long)
        empty_loader = fabric.setup_dataloaders(
            create_dataloader(
                TensorDataset(inputs[:0].cpu(), empty_labels, empty_labels),
                args.batch_size,
                num_gpus=fabric.world_size,
                num_workers=0,
                is_training=False,
                batch_size_mode=args.batch_size_mode,
            )
        )
        empty_checks = [
            lambda: evaluate(fabric, first, empty_loader, do_global_aggregation=True),
            lambda: ZRF(fabric, first, second, empty_loader),
            lambda: actv_dist(fabric, first, second, empty_loader),
            lambda: calculate_completeness(fabric, first, second, empty_loader),
            lambda: get_membership_attack_prob(
                fabric,
                fabric.world_size,
                first,
                empty_loader,
                empty_loader,
                empty_loader,
            ),
        ]
        for empty_check in empty_checks:
            try:
                empty_check()
            except ValueError:
                pass
            else:
                raise AssertionError("Empty evaluation dataset was not rejected")

        distance = model_lay_dist(fabric, first, second)["metric_value_dict"][
            "final_value"
        ]
        assert abs(distance - expected_distance) < 1e-6
        assert (
            model_lay_dist(fabric, first, first)["metric_value_dict"]["final_value"]
            == 0
        )
        # Summoning full parameters must not break subsequent inference.
        evaluate(fabric, first, loader, do_global_aggregation=True)
        fabric.barrier()
        if args.fail_rank is not None:
            if fabric.global_rank == args.fail_rank:
                raise RuntimeError("Injected Stage 3 rank-local failure")
            fabric.barrier()  # supervisor must terminate this waiting peer
        print(f"STAGE3_PROBE_PASS rank={fabric.global_rank}", flush=True)
        fabric.barrier()
        torch.distributed.destroy_process_group()
    except Exception as exc:
        handle_distributed_error(fabric, exc)


if __name__ == "__main__":
    main()
