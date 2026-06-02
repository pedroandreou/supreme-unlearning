import os
import glob
import argparse
import sys
import supreme.utils.project_config as project_config


def print_banner(message):
    """Prints a banner with a message."""
    print("\n" + "#" * 80)
    print(f"### {message.center(72)} ###")
    print("#" * 80 + "\n")


def main():
    """
    Main function to find the latest model checkpoints.
    This script is designed to find the absolute path to the latest and best-performing
    model checkpoint based on a set of input parameters.
    """
    parser = argparse.ArgumentParser(
        description="Find the latest model checkpoint path."
    )
    parser.add_argument(
        "-precision",
        type=str,
        required=True,
        help="Precision used during training (e.g., '32-true')",
    )
    parser.add_argument(
        "-training_seed",
        type=str,
        required=True,
        help="Seed used for reproducible training (or 'none')",
    )
    parser.add_argument(
        "-unlearning_seed", type=str, required=True, help="Seed used for unlearning"
    )
    parser.add_argument(
        "-num_gpus",
        type=int,
        required=True,
        help="Number of GPUs used, for path construction.",
    )
    parser.add_argument(
        "-include_gpus_in_path",
        type=str,
        default="true",
        help="Flag to include GPU count in the checkpoint path.",
    )
    parser.add_argument(
        "-distributed_strategy",
        type=str,
        default="ddp",
        help="Distributed strategy used (ddp, fsdp, deepspeed). Used for path construction.",
    )
    parser.add_argument(
        "-deepspeed_stage",
        type=int,
        default=2,
        help="DeepSpeed ZeRO stage (1, 2, or 3). Used for path construction when strategy is deepspeed.",
    )
    parser.add_argument(
        "-net",
        type=str,
        required=True,
        help="Specific model to find (e.g., 'ResNet18', 'ViT').",
    )
    parser.add_argument(
        "-dataset",
        type=str,
        required=True,
        help="Specific dataset to find (e.g., 'Cifar20', 'Cifar100').",
    )
    args = parser.parse_args()

    # Replicate the checkpoint path structure from the training script
    training_seed_str = f"train_seed_{args.training_seed}"
    unlearning_seed_str = f"unlearning_seed_{args.unlearning_seed}"
    gpu_str = f"{args.num_gpus}gpus" if args.include_gpus_in_path == "true" else ""
    if args.num_gpus > 1:
        if args.distributed_strategy == "deepspeed":
            dist_str = f"dist_deepspeed_stage{args.deepspeed_stage}"
        else:
            dist_str = f"dist_{args.distributed_strategy}"
    else:
        dist_str = "no_dist"

    base_checkpoint_path = os.path.join(
        project_config.CHECKPOINT_PATH,
        f"precision_{args.precision}",
        gpu_str,
        dist_str,
        training_seed_str,
        unlearning_seed_str,
        "model_checkpoints",
        args.net,
        args.dataset,
    )

    if not os.path.isdir(base_checkpoint_path):
        # This is not an error, it just means no checkpoint exists yet.
        # The calling script will handle this case.
        sys.exit(1)

    # Find all timestamped directories for this model/dataset
    search_path_pattern = os.path.join(base_checkpoint_path, "*")
    time_dirs = [p for p in glob.glob(search_path_pattern) if os.path.isdir(p)]

    if not time_dirs:
        sys.exit(1)

    # Only trust dirs where training actually finished - proven by the
    # TRAINING_DONE marker written by MAIN_scaled.sh after train_model returns
    # successfully. Without this filter, a partial best.pth from an in-progress
    # run can be picked up by another task waiting on the training lock.
    time_dirs = [
        d for d in time_dirs if os.path.isfile(os.path.join(d, "TRAINING_DONE"))
    ]
    if not time_dirs:
        sys.exit(1)

    all_checkpoints = []
    for a_dir in time_dirs:
        all_checkpoints.extend(glob.glob(os.path.join(a_dir, "*-best.pth")))

    if not all_checkpoints:
        sys.exit(1)

    # Find the checkpoint with the highest epoch number from its filename
    latest_checkpoint = max(
        all_checkpoints,
        key=lambda x: int(os.path.basename(x).split("-")[2]),
    )

    # Print the absolute path to the best checkpoint
    print(os.path.abspath(latest_checkpoint))


if __name__ == "__main__":
    main()
