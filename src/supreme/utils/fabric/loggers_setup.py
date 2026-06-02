"""Build Lightning Fabric loggers (CSV, TensorBoard) from a config dict.

These are passed to Fabric at construction time via the `loggers=` argument.
The WandbLogger is wired up separately in `wandb_utils/runtime/wandb_setup.py`
because it requires authentication and must be set up post-init.
"""

from typing import List


def build_fabric_loggers(config) -> List:
    """Return a list of Fabric loggers based on flags in `config`.

    Recognised keys:
        tensorboard_logging_flag: bool - instantiate TensorBoardLogger.
        csv_logging_flag: bool - instantiate CSVLogger.
        logging_root_dir: str - root directory for log files.
        logging_run_name: str - sub-directory name under root_dir.

    Returns an empty list if no flags are set. Raises ImportError with a clear
    install hint if TensorBoard is requested but the backing package is missing.
    """
    loggers: List = []

    if config.get("csv_logging_flag"):
        from lightning.fabric.loggers import CSVLogger

        loggers.append(
            CSVLogger(
                root_dir=config["logging_root_dir"],
                name=config.get("logging_run_name", "fabric_run"),
            )
        )

    if config.get("tensorboard_logging_flag"):
        from lightning.fabric.loggers import TensorBoardLogger

        try:
            loggers.append(
                TensorBoardLogger(
                    root_dir=config["logging_root_dir"],
                    name=config.get("logging_run_name", "fabric_run"),
                )
            )
        except ModuleNotFoundError as e:
            raise ImportError(
                "TensorBoardLogger requires the 'tensorboard' (or 'tensorboardX') package. "
                "Install with: pip install tensorboard"
            ) from e

    return loggers
