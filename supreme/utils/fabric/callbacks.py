"""
See the tutorial of Callbacks at:
https://lightning.ai/docs/pytorch/LTS/fabric/guide/callbacks.html
"""

import time
from lightning.fabric import Fabric
from typing import Optional

# from lightning.pytorch.callbacks import ModelCheckpoint

# checkpoint_callback = ModelCheckpoint(
#     monitor="val_accuracy",
#     mode="max",
#     filename="{epoch:02d}-{val_accuracy:.4f}",
#     save_top_k=3,
#     auto_insert_metric_name=False,
# )


class TrainingCallback:
    """Callback for training metrics and timing"""

    def __init__(self, logging_enabled: bool = False):
        self.logging_enabled = logging_enabled
        self.fabric: Optional[Fabric] = None
        self.batch_start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None

    def on_train_epoch_start(self, fabric: Fabric, epoch: int):
        self.fabric = fabric
        self.epoch_start_time = time.time()

        self.fabric.print(f"\nTraining epoch {epoch} started")

    def on_train_batch_start(self):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, loss, epoch, batch_idx, lr):
        if self.fabric is None or self.batch_start_time is None:
            return
        batch_time = time.time() - self.batch_start_time

        if batch_idx % 10 == 0:
            self.fabric.print(f"Logging batch {batch_idx}")

            metrics = {
                "train/epoch": epoch,
                "train/batch": batch_idx,
                "train/loss": loss.item(),
                "train/learning_rate": lr,
                "train/batch_time": batch_time,
            }

            if self.logging_enabled:
                self.fabric.log_dict(metrics)
            else:
                self.fabric.print(
                    f"Training: "
                    f"Epoch [{epoch}], "
                    f"Batch [{batch_idx}], "
                    f"Loss: {loss.item():.4f}, "
                    f"LR: {lr:.5f}, "
                    f"Time: {batch_time:.2f}s"
                )

    def on_train_epoch_end(self, epoch, train_loss, last_lr):
        if self.fabric is None or self.epoch_start_time is None:
            return
        epoch_time = time.time() - self.epoch_start_time

        metrics = {
            "train/epoch": epoch,
            "train/loss": train_loss,
            "train/learning_rate": last_lr,
            "train/epoch_time": epoch_time,
        }
        if self.logging_enabled:
            self.fabric.log_dict(metrics)
        else:
            self.fabric.print(
                "Training Epoch [{train/epoch}], "
                "LR: {train/learning_rate:.5f}, "
                "Loss: {train/loss:.4f}, "
                "Time: {train/epoch_time:.2f}s".format(**metrics)
            )


class TestCallback:
    """Callback for test metrics"""

    def __init__(self, logging_enabled: bool = False):
        self.logging_enabled = logging_enabled
        self.fabric: Optional[Fabric] = None
        self.val_start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None

    def on_test_epoch_start(self, fabric: Fabric):
        self.fabric = fabric
        self.val_start_time = time.time()

    def on_test_batch_start(self):
        self.batch_start_time = time.time()

    def on_test_batch_end(self, loss, epoch, batch_idx, acc):
        if self.fabric is None or self.batch_start_time is None:
            return
        batch_time = time.time() - self.batch_start_time

        if batch_idx % 100 == 0:
            self.fabric.print(f"Logging batch {batch_idx}")

            metrics = {
                "val/epoch": epoch,
                "val/batch": batch_idx,
                "val/loss": loss.item(),
                "val/accuracy": acc,
                "val/batch_time": batch_time,
            }
            if self.logging_enabled:
                self.fabric.log_dict(metrics)
            else:
                self.fabric.print(
                    f"Validation: "
                    f"Epoch [{epoch}], "
                    f"Batch [{batch_idx}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {acc:.4f}, "
                    f"Time: {batch_time:.2f}s"
                )

    def on_test_epoch_end(self, epoch, loss, acc):
        if self.fabric is None or self.val_start_time is None:
            return
        val_time = time.time() - self.val_start_time

        metrics = {
            "val/epoch": epoch,
            "val/loss": loss,
            "val/accuracy": acc,
            "val/time": val_time,
        }
        if self.logging_enabled:
            self.fabric.log_dict(metrics)
        else:
            self.fabric.print(
                "Validation: "
                "Epoch: {val/epoch}, "
                "Loss: {val/loss:.4f}, "
                "Accuracy: {val/accuracy:.4f}, "
                "Time: {val/time:.2f}s".format(**metrics)
            )


class ParameterModificationCallback:
    """Callback for parameter modification metrics specifically for the SSD unlearning method"""

    def __init__(self, logging_enabled: bool = False):
        self.logging_enabled = logging_enabled
        self.fabric: Optional[Fabric] = None
        self.mod_start_time: Optional[float] = None
        self.layer_start_time: Optional[float] = None

    def on_modification_epoch_start(self, fabric: Fabric):
        self.fabric = fabric
        self.mod_start_time = time.time()

    def on_modification_batch_start(self):
        self.layer_start_time = time.time()

    def on_modification_batch_end(
        self,
        layer_name: str,
        layer_modified_params: int,
        layer_total_params: int,
        layer_idx: int,
        total_layers: int,
    ):
        if self.fabric is None or self.layer_start_time is None:
            return
        layer_time = time.time() - self.layer_start_time

        if layer_idx % 10 == 0:  # Log every 20 layers
            self.fabric.print(f"Logging batch {layer_idx}")

            metrics = {
                "param_mod/layer": layer_idx,
                "param_mod/layer_name": layer_name,
                "param_mod/layer_modified": layer_modified_params,
                "param_mod/layer_total": layer_total_params,
                "param_mod/layer_ratio": layer_modified_params / layer_total_params,
                "param_mod/layer_time": layer_time,
            }

            if self.logging_enabled:
                self.fabric.log_dict(metrics)
            else:
                self.fabric.print(
                    f"Layer Modification [{layer_idx}/{total_layers}] {layer_name}: "
                    f"Modified: {layer_modified_params}, "
                    f"Total: {layer_total_params}, "
                    f"Ratio: {layer_modified_params/layer_total_params:.4f}, "
                    f"Time: {layer_time:.2f}s"
                )

    def on_modification_epoch_end(self, num_modified_params: int, total_params: int):
        if self.fabric is None or self.mod_start_time is None:
            return
        mod_time = time.time() - self.mod_start_time

        metrics = {
            "param_mod/total_modified": num_modified_params,
            "param_mod/total_params": total_params,
            "param_mod/final_ratio": num_modified_params / total_params,
            "param_mod/total_time": mod_time,
        }
        if self.logging_enabled:
            self.fabric.log_dict(metrics)
        else:
            self.fabric.print(
                "Final Parameter Modification: "
                f"Modified Parameters: {num_modified_params}, "
                f"Total Parameters: {total_params}, "
                f"Modification Ratio: {num_modified_params/total_params:.4f}, "
                f"Time: {mod_time:.2f}s"
            )


class MetricsEvaluationCallback:
    """Callback for evaluation metrics tracking"""

    def __init__(self, logging_enabled: bool = False):
        self.logging_enabled = logging_enabled
        self.fabric: Optional[Fabric] = None
        self.epoch_start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None
        self.metric_name: Optional[str] = None

    def on_evaluation_epoch_start(self, fabric: Fabric, epoch: int, metric_name: str):
        self.fabric = fabric
        self.epoch_start_time = time.time()
        self.metric_name = metric_name
        self.fabric.print(f"\nEvaluating {self.metric_name}...")

    def on_evaluation_batch_start(self):
        self.batch_start_time = time.time()

    def on_evaluation_batch_end(
        self,
        batch_idx: int,
        epoch: int,
        batch_value: Optional[float] = None,
    ):
        if (
            self.fabric is None
            or self.batch_start_time is None
            or self.metric_name is None
        ):
            return
        batch_time = time.time() - self.batch_start_time

        if batch_idx % 10 == 0:  # Log every 20 batches
            self.fabric.print(f"Processing batch {batch_idx}")

            metrics = {
                f"metrics/{self.metric_name}/epoch": epoch,
                f"metrics/{self.metric_name}/batch": batch_idx,
                f"metrics/{self.metric_name}/batch_time": batch_time,
            }

            if batch_value is not None:
                metrics[f"metrics/{self.metric_name}/batch_value"] = batch_value
            else:
                print(f"The batch values is None for batch {batch_idx}")

            if self.logging_enabled:
                self.fabric.log_dict(metrics)
            else:
                value_str = (
                    f", Value: {batch_value:.4f}" if batch_value is not None else ""
                )
                self.fabric.print(
                    f"Evaluation: "
                    f"Epoch [{epoch}], "
                    f"Batch [{batch_idx}]{value_str}, "
                    f"Time: {batch_time:.2f}s"
                )

    def on_evaluation_epoch_end(self, epoch: int, epoch_value: Optional[float] = None):
        if (
            self.fabric is None
            or self.epoch_start_time is None
            or self.metric_name is None
        ):
            return
        epoch_time = time.time() - self.epoch_start_time

        metrics = {
            f"metrics/{self.metric_name}/epoch": epoch,
            f"metrics/{self.metric_name}/epoch_time": epoch_time,
        }

        if epoch_value is not None:
            metrics[f"metrics/{self.metric_name}/epoch_value"] = epoch_value

        if self.logging_enabled:
            self.fabric.log_dict(metrics)
        else:
            self.fabric.print(
                f"\n{self.metric_name} Results:"
                f"{' Value: ' + f'{epoch_value:.4f}' if epoch_value is not None else ''}"
                f" (Total time: {epoch_time:.2f}s)\n"
            )
