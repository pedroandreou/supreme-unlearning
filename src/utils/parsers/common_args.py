import argparse
import os


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="net type")
    parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument(
        "-wandb_logging_flag",
        action="store_true",
        default=False,
        help="Enable logging to wandb (default: False)",
    )
    parser.add_argument(
        "-tensorboard_logging_flag",
        action="store_true",
        default=False,
        help="Enable Lightning Fabric TensorBoardLogger (requires `tensorboard` or `tensorboardX` installed). Default: False",
    )
    parser.add_argument(
        "-csv_logging_flag",
        action="store_true",
        default=False,
        help="Enable Lightning Fabric CSVLogger (writes metrics.csv under -logging_root_dir). Default: False",
    )
    parser.add_argument(
        "-logging_root_dir",
        type=str,
        default=os.environ.get("FABRIC_LOGGING_ROOT_DIR", "./fabric_logs"),
        help="Root directory for Fabric CSV/TensorBoard loggers (default: ./fabric_logs or FABRIC_LOGGING_ROOT_DIR env var)",
    )
    parser.add_argument(
        "-export_class_distribution_info_flag",
        action="store_true",
        default=False,
        help="Enable logging to wandb (default: False)",
    )
    parser.add_argument(
        "-use_process_tracker",
        action="store_true",
        default=False,
        help="Enable process tracker (default: False)",
    )
    parser.add_argument(
        "-distributed_strategy",
        type=str,
        default=os.environ.get("DISTRIBUTED_STRATEGY", "ddp"),
        choices=["ddp", "fsdp", "deepspeed", "auto", "xla"],
        help="Distributed training strategy: ddp, fsdp, deepspeed, auto, or xla (default: ddp, or DISTRIBUTED_STRATEGY env var)",
    )
    parser.add_argument(
        "-deepspeed_stage",
        type=int,
        default=int(os.environ.get("DEEPSPEED_STAGE", "2")),
        choices=[1, 2, 3],
        help="DeepSpeed ZeRO stage: 1 (optimizer sharding), 2 (optimizer+gradient sharding), 3 (full parameter sharding). Only used when -distributed_strategy is deepspeed. (default: 2, or DEEPSPEED_STAGE env var)",
    )
    return parser
