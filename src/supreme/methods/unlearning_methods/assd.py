"""ASSD (Adaptive Selective Synaptic Dampening) unlearning method.

Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" (https://arxiv.org/abs/2308.07707)

Notes:
Adaptive variant of SSD, with adaptive parameter selection (from error_unlearning.ipynb).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import wandb
from lightning.fabric import Fabric
from typing import Dict, List, Optional, Sequence


class ParameterPerturber:
    def __init__(
        self,
        fabric: Fabric,
        model: nn.Module,
        opt: torch.optim.Optimizer,
        parameters: Optional[Dict] = None,
    ):
        self.fabric = fabric
        self.model = model
        self.opt = opt

        if parameters is None:
            raise ValueError(
                "Parameters dictionary cannot be None for ParameterPerturber"
            )

        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]  # unused
        self.max_layer = parameters["max_layer"]  # unused
        self.forget_threshold = parameters["forget_threshold"]  # unused
        self.dampening_constant = parameters["dampening_constant"]  # Lambda from paper
        self.selection_weighting = parameters[
            "selection_weighting"
        ]  # Alpha from paper (will be adaptive)
        self.adaptive_percentile = parameters.get(
            "adaptive_percentile", True
        )  # Enable adaptive selection

    def get_layer_num(self, layer_name: str) -> int:
        """
        This method extracts and returns the numerical ID of a layer from its name.
        This can be useful for targeting specific layers for modifications or analysis.
        """

        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter values
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict([(k, torch.zeros_like(p)) for k, p in model.named_parameters()])

    def fulllike_params_dict(
        self, model: nn.Module, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: Sequence[int]) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape))
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: Dataset, sample_perc: float) -> Subset:
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): percentage of dataset to sample. range(0,1)
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc)))  # type: ignore
        return Subset(dataset, sample_idxs)  # type: ignore

    def split_dataset_by_class(self, dataset: Dataset) -> List[Subset]:
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        n_classes = len(set([target for _, target in dataset]))  # type: ignore
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset):  # type: ignore
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]  # type: ignore

    def calc_importance(
        self, dataloader: DataLoader, criterion
    ) -> Dict[str, torch.Tensor]:
        """
        Adapted from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """

        # Initializes a dictionary of the same structure as the model's parameters but with all values set to zero
        # This dictionary will store the importance values
        importances = self.zerolike_params_dict(self.model)

        # # Call test start since this is an evaluation phase
        # if self.fabric.global_rank == 0:
        #     self.fabric.call("on_test_epoch_start", fabric=self.fabric)
        # self.fabric.barrier()

        for batch_idx, batch in enumerate(dataloader):
            # if self.fabric.global_rank == 0:
            #     self.fabric.call("on_test_batch_start")
            # self.fabric.barrier()

            x, _, y = batch
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            # Use fabric.backward() for gradient handling (mixed precision, sync).
            # For non-DDP strategies (parameter-surgery methods skip wrapping),
            # fall back to loss.backward().
            if hasattr(self.model, "_forward_module") or hasattr(self.model, "module"):
                self.fabric.backward(loss)
            else:
                loss.backward()

            # k1: The name of the parameter in the model.
            # p: The parameter tensor itself.
            # k2: The name of the parameter in the importances dictionary.
            # imp: The tensor in the importances dictionary that corresponds to the importance of the parameter.
            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                # For each parameter p, if gradient is not None
                if p.grad is not None:
                    # it accumulates the squared gradient
                    # and adds it to the importance value
                    imp.data += p.grad.data.clone().pow(2)

            # if self.fabric.global_rank == 0:
            #     self.fabric.call(
            #         "on_test_batch_end",
            #         loss=loss,
            #         epoch=batch_idx // len(dataloader),
            #         batch_idx=batch_idx,
            #         acc=0,
            #     )
            # self.fabric.barrier()

        # Average over mini batch length (number of batches on this GPU)
        # This computes the mean squared gradient per batch locally
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))  # type: ignore

        # Aggregate the per-batch averages across all GPUs using mean operation
        # The default reduce_op='mean' averages the per-batch averages from all processes,
        # maintaining the same semantic meaning as the original single-GPU implementation
        for imp in importances.values():
            self.fabric.all_reduce(imp.data)  # Default reduce_op='mean'

        return importances

    def compute_adaptive_percentile(
        self,
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
        forget_set_size: int,
        total_set_size: int,
    ) -> float:
        """
        Compute adaptive percentile threshold based on importance ratio distribution.
        This implements the adaptive selection logic from error_unlearning.ipynb.

        Parameters:
        original_importance: Importance values for the full dataset
        forget_importance: Importance values for the forget set
        forget_set_size: Number of samples in forget set
        total_set_size: Total number of samples (forget + retain)

        Returns:
        percentile: The computed percentile threshold value
        """

        # Collect all relative importance values (forget/original ratio)
        all_relative_values = []

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Calculate ratio of forget importance to original importance
                divs_ = fimp.div(oimp)

                # Remove NaN and Inf values
                divs_ = divs_[~torch.isnan(divs_)]
                divs_ = divs_[~torch.isinf(divs_)]

                if divs_.numel() > 0:
                    all_relative_values.append(divs_.reshape(-1).cpu().numpy())

        # Concatenate all relative values
        if len(all_relative_values) > 0:
            all_relative_values = np.concatenate(all_relative_values)
        else:
            self.fabric.print("WARNING: No valid relative importance values found!")
            return 99.0  # Default fallback

        # Calculate percentile based on dataset sizes
        # This uses logarithmic scaling to determine the percentile
        len_forget = forget_set_size
        len_all = total_set_size

        # Adaptive percentile calculation from the notebook
        share_off = np.log(1 + (len_forget / len_all) * 100)
        percentile_value = 100 - share_off

        self.fabric.print(f"Forget set size: {len_forget}, Total size: {len_all}")
        self.fabric.print(f"Computed adaptive percentile: {percentile_value:.2f}")

        # Calculate the actual threshold value at this percentile
        # Compute on rank 0 and broadcast to ensure consistency across all GPUs
        if self.fabric.global_rank == 0:
            threshold = float(np.nanpercentile(all_relative_values, percentile_value))
            threshold_tensor = torch.tensor([threshold], device=self.fabric.device)
        else:
            threshold_tensor = torch.zeros(1, device=self.fabric.device)

        threshold_tensor = self.fabric.broadcast(threshold_tensor, src=0)
        threshold = threshold_tensor.item()
        self.fabric.print(f"Adaptive selection threshold value: {threshold:.6f}")

        return threshold

    def modify_weight(
        self,
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
        forget_set_size: int = None,
        total_set_size: int = None,
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper.
        If adaptive_percentile is enabled, computes the selection_weighting dynamically.

        Parameters:
        original_importance: Importance values for original dataset
        forget_importance: Importance values for forget sample
        forget_set_size: Number of samples in forget set (required for adaptive mode)
        total_set_size: Total number of samples (required for adaptive mode)

        Returns:
        None
        """

        # Compute adaptive selection weighting if enabled
        if self.adaptive_percentile:
            if forget_set_size is None or total_set_size is None:
                raise ValueError(
                    "forget_set_size and total_set_size must be provided for adaptive percentile calculation"
                )

            # Compute and update selection_weighting based on importance distribution
            self.selection_weighting = self.compute_adaptive_percentile(
                original_importance,
                forget_importance,
                forget_set_size,
                total_set_size,
            )

            self.fabric.print(
                f"Using adaptive selection_weighting (alpha): {self.selection_weighting:.6f}"
            )
        else:
            self.fabric.print(
                f"Using fixed selection_weighting (alpha): {self.selection_weighting:.6f}"
            )

        # if self.fabric.global_rank == 0:
        #     self.fabric.call("on_modification_epoch_start", fabric=self.fabric)
        # self.fabric.barrier()

        total_modified = 0
        total_params = 0
        # total_layers = len(list(self.model.named_parameters()))

        with torch.no_grad():
            # (n, p): Represents the current parameter name and tensor from the model.
            # (oimp_n, oimp): Represents the current parameter name and its corresponding importance value from original_importance dictionary.
            # (fimp_n, fimp): Represents the current parameter name and its corresponding importance value from forget_importance dictionary.
            for batch_idx, ((n, p), (oimp_n, oimp), (fimp_n, fimp)) in enumerate(
                zip(
                    self.model.named_parameters(),
                    original_importance.items(),
                    forget_importance.items(),
                )
            ):
                # if self.fabric.global_rank == 0:
                #     self.fabric.call("on_modification_batch_start")
                # self.fabric.barrier()

                layer_total = p.numel()

                #########################################################
                # Synapse Selection with parameter alpha (selection_weighting)
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Count number of modified parameters for this layer
                layer_modified = torch.sum(fimp > oimp_norm)

                # Weight modification
                # Synapse Dampening with parameter lambda (dampening_constant)
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
                #########################################################

                # Update totals
                total_modified += layer_modified
                total_params += layer_total

                # if self.fabric.global_rank == 0:
                #     self.fabric.call(
                #         "on_modification_batch_end",
                #         layer_name=n,
                #         layer_modified_params=layer_modified.item(),
                #         layer_total_params=layer_total,
                #         layer_idx=batch_idx,
                #         total_layers=total_layers,
                #     )
                # self.fabric.barrier()

        # # Final reduction for totals
        # total_modified = self.fabric.all_gather(total_modified).sum().float().mean()  # type: ignore
        # total_params = self.fabric.all_gather(total_params).sum().float().mean()  # type: ignore

        # if self.fabric.global_rank == 0:
        #     self.fabric.call(
        #         "on_modification_epoch_end",
        #         num_modified_params=total_modified.item(),
        #         total_params=total_params.item(),
        #     )
        # self.fabric.barrier()

        # Log modification statistics
        if self.fabric.global_rank == 0:
            modification_rate = (total_modified.float() / total_params) * 100
            self.fabric.print(
                f"Modified {total_modified}/{total_params} parameters ({modification_rate:.2f}%)"
            )


###############################################


def assd(  # Adaptive Selective Synaptic Dampening
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,  # Independent copy of the original model so it can go under the unlearning procedure
    forget_train_dataloader: DataLoader,
    full_train_dataloader: DataLoader,
    dampening_constant: float,
    selection_weighting: float,  # This will be used as initial value, then adapted
    **kwargs,
):
    """
    Adaptive Selective Synaptic Dampening (ASSD) for machine unlearning.

    This method extends SSD by automatically adapting the selection_weighting (alpha)
    parameter based on the importance ratio distribution and dataset characteristics.

    Parameters:
    fabric: Lightning Fabric instance
    num_gpus: Number of GPUs
    wandb_logging_flag: Whether to log to wandb
    model: Model to unlearn from
    forget_train_dataloader: DataLoader for forget set
    full_train_dataloader: DataLoader for full training set
    dampening_constant: Lambda parameter from SSD paper
    selection_weighting: Initial alpha parameter (will be overridden by adaptive computation)
    """

    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # unused
        "max_layer": -1,  # unused
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Initial alpha (will be adapted)
        "adaptive_percentile": True,  # Enable adaptive selection
    }

    raw_model = model.module if hasattr(model, "module") else model
    distributed_strategy_name = kwargs.get("distributed_strategy_name", "ddp")

    # Parameter-surgery methods (SSD family) skip FSDP wrapping - see ssd.py for rationale.
    if distributed_strategy_name != "ddp":
        model = raw_model.to(fabric.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    else:
        optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.1)
        model, optimizer = fabric.setup(raw_model, optimizer)  # type: ignore

    criterion = nn.CrossEntropyLoss(reduction="mean")

    if fabric.global_rank == 0 and wandb_logging_flag:
        # Update the config with ASSD-specific parameters
        config_dict = {
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "learning_rate": 0.1,
            "dampening_constant": dampening_constant,
            "selection_weighting_initial": selection_weighting,
            "adaptive_percentile": True,
            "lower_bound": parameters["lower_bound"],
            "exponent": parameters["exponent"],
        }
        wandb.config.update(config_dict)
    fabric.barrier()

    pdr = ParameterPerturber(fabric, model, optimizer, parameters)
    model = model.eval()

    # Calculation of the forget set importances
    fabric.print("Calculating forget importances...")
    forget_importances = pdr.calc_importance(forget_train_dataloader, criterion)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    fabric.print("Calculating original importances...")
    original_importances = pdr.calc_importance(full_train_dataloader, criterion)

    # Get dataset sizes for adaptive calculation
    forget_set_size = len(forget_train_dataloader.dataset)  # type: ignore
    total_set_size = len(full_train_dataloader.dataset)  # type: ignore

    # Debug: Check importance statistics for first few layers
    if fabric.global_rank == 0:
        fabric.print("=" * 80)
        fabric.print("ASSD IMPORTANCE STATISTICS:")
        for idx, (name, fimp) in enumerate(forget_importances.items()):
            if idx >= 3:  # Only first 3 layers
                break
            oimp = original_importances[name]
            fabric.print(f"\n{name}:")
            fabric.print(
                f"  Forget  - min: {fimp.min().item():.8f}, max: {fimp.max().item():.8f}, mean: {fimp.mean().item():.8f}"
            )
            fabric.print(
                f"  Original - min: {oimp.min().item():.8f}, max: {oimp.max().item():.8f}, mean: {oimp.mean().item():.8f}"
            )
        fabric.print("=" * 80)

    # Dampen selected parameters with adaptive selection weighting
    fabric.print("Computing adaptive parameters and applying weight dampening...")
    pdr.modify_weight(
        original_importances,
        forget_importances,
        forget_set_size=forget_set_size,
        total_set_size=total_set_size,
    )

    # Log final selection weighting to wandb
    if fabric.global_rank == 0 and wandb_logging_flag:
        wandb.config.update({"selection_weighting_final": pdr.selection_weighting})

    fabric.print("ASSD unlearning complete.")

    # Return the post-fabric.setup() wrapped model (see ssd.py for rationale).
    return model
