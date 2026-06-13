"""LFSSD-Det (Deterministic Loss-Free Selective Synaptic Dampening) unlearning method.

Paper: "LOSS-FREE MACHINE UNLEARNING" (https://arxiv.org/pdf/2402.19308)
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/main/README.md

Notes:
Deterministic variant of LFSSD (see lfssd.py) that computes true per-sample
gradients, analogous to ssd_det.py vs ssd.py.

The original LFSSD implementation uses .mean() over the batch before backward, so
gradients are averaged across the batch before taking abs(). Because abs() is
applied after the mean, different batch compositions (from shuffling) produce
mathematically different importance scores: mean([a, b]) != mean([a, c]).

This variant computes the per-sample loss and backpropagates each individually,
then accumulates abs(g_i). This is order-independent and matches the per-sample
spirit of the loss-free formulation.
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
        self.selection_weighting = parameters["selection_weighting"]  # Alpha from paper

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
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
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
        self,
        dataloader: DataLoader,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate per-parameter importance using true per-sample gradients.

        Unlike the standard LFSSD calc_importance which computes the gradient of the
        batch-mean loss (order-dependent), this computes Σᵢ |gᵢ| by backwarding each
        per-sample loss (‖f(xᵢ)‖²_F) individually. This is order-independent.

        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary
            containing importance for each parameter
        """

        importances = self.zerolike_params_dict(self.model)
        total_samples = 0

        for batch_idx, batch in enumerate(dataloader):
            x, _, y = batch
            out = self.model(x)
            # Per-sample loss: ‖f(xᵢ)‖²_F for each sample i, shape: [batch_size]
            losses = torch.norm(out, p="fro", dim=1).pow(2)

            # Backward each per-sample loss individually to get true per-sample gradients
            for i in range(len(losses)):
                self.opt.zero_grad()
                # Only retain graph if there are more samples to process in this batch
                retain = i < len(losses) - 1
                if hasattr(self.model, "_forward_module") or hasattr(
                    self.model, "module"
                ):
                    self.fabric.backward(losses[i], retain_graph=retain)
                else:
                    losses[i].backward(retain_graph=retain)

                for (k1, p), (k2, imp) in zip(
                    self.model.named_parameters(), importances.items()
                ):
                    if p.grad is not None:
                        imp.data += p.grad.data.clone().abs()

                total_samples += 1

        # Average over total number of samples (not number of batches)
        for _, imp in importances.items():
            imp.data /= float(total_samples)

        # Aggregate across all GPUs
        for imp in importances.values():
            self.fabric.all_reduce(imp.data)  # Default reduce_op='mean'

        return importances

    def modify_weight(
        self,
        original_importance: Dict[str, torch.Tensor],
        forget_importance: Dict[str, torch.Tensor],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample

        Returns:
        None

        """

        total_modified = 0
        total_params = 0

        with torch.no_grad():
            for batch_idx, ((n, p), (oimp_n, oimp), (fimp_n, fimp)) in enumerate(
                zip(
                    self.model.named_parameters(),
                    original_importance.items(),
                    forget_importance.items(),
                )
            ):
                layer_total = p.numel()

                #########################################################
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Count number of modified parameters for this layer
                layer_modified = torch.sum(fimp > oimp_norm)

                # Weight modification
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )  # Synapse Dampening with parameter lambda
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
                #########################################################

                # Update totals
                total_modified += layer_modified
                total_params += layer_total


###############################################


def lfssd_det(
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,
    forget_train_dataloader: DataLoader,
    full_train_dataloader: DataLoader,
    dampening_constant: float,
    selection_weighting: float,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # unused
        "max_layer": -1,  # unused
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
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

    if fabric.global_rank == 0 and wandb_logging_flag:
        config_dict = {
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": 0.1,
            "dampening_constant": dampening_constant,
            "selection_weighting": selection_weighting,
            "lower_bound": parameters["lower_bound"],
            "exponent": parameters["exponent"],
            "deterministic_importance": True,
        }
        wandb.config.update(config_dict)
    fabric.barrier()

    pdr = ParameterPerturber(fabric, model, optimizer, parameters)
    model = model.eval()

    # Calculation of the forget set importances
    forget_importances = pdr.calc_importance(forget_train_dataloader)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dataloader)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, forget_importances)

    # Return the post-fabric.setup() wrapped model (see ssd.py for rationale).
    return model
