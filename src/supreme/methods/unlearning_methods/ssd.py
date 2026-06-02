"""SSD (Selective Synaptic Dampening) unlearning method.

Paper: "Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening" (https://arxiv.org/abs/2308.07707)
Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/75fdea18497b0f5d654b136753a386fe74b9cd26/src/ssd.py#L35
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
        # self.alpha = None
        # self.xmin = None

        if parameters is None:
            raise ValueError(
                "Parameters dictionary cannot be None for ParameterPerturber"
            )

        # self.fabric.print(parameters)
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
        self, dataloader: DataLoader, criterion
    ) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
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

        # test_loss = 0.0
        # correct = 0.0

        for batch_idx, batch in enumerate(dataloader):
            # if self.fabric.global_rank == 0:
            #     self.fabric.call("on_test_batch_start")
            # self.fabric.barrier()

            x, _, y = batch
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            # Use fabric.backward() for gradient handling (mixed precision, sync).
            # For FSDP-unwrapped models (parameter-surgery methods skip FSDP wrapping
            # to avoid NCCL deadlocks), fall back to loss.backward().
            if hasattr(self.model, "_forward_module") or hasattr(self.model, "module"):
                self.fabric.backward(loss)
            else:
                loss.backward()

            # #########################################################
            # test_loss += loss.item()
            # _, preds = out.max(1)
            # correct += preds.eq(y).sum()
            # #########################################################

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
            #         acc=100 * correct.float() / y.size(0),
            #     )
            # self.fabric.barrier()

        # # Aggregate metrics across processes
        # test_loss = self.fabric.all_gather(test_loss).sum()  # type: ignore
        # correct = self.fabric.all_gather(correct).sum()  # type: ignore

        # Calculate final metrics
        # avg_loss = test_loss / len(dataloader.dataset)
        # final_acc = 100 * correct.float() / len(dataloader.dataset)

        # if self.fabric.global_rank == 0:
        #     # Call test end with proper metrics
        #     self.fabric.call(
        #         "on_test_epoch_end",
        #         epoch=batch_idx // len(dataloader),
        #         loss=avg_loss,
        #         acc=final_acc,
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
        # threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

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


###############################################


def ssd(  # also named as pdr_tuning
    fabric: Fabric,
    num_gpus: int,
    wandb_logging_flag: bool,
    model: nn.Module,  # Independent copy of the original model so it can go under the unlearning procedure
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

    # SSD does not do gradient-descent training - it computes importance scores
    # via forward/backward and then directly modifies weights in-place. Both FSDP
    # and DeepSpeed cause issues with this pattern:
    # - FSDP: implicit all-gather/reduce-scatter during forward/backward cause NCCL
    #   deadlocks because SSD's per-parameter iteration doesn't match FSDP's
    #   expected collective sequence.
    # - DeepSpeed: in-place parameter modifications via modify_weight() don't persist
    #   correctly on DeepSpeed-wrapped models, causing the model to retain the
    #   forgotten class (forget_acc ~80-97% instead of ~0%).
    # Since SSD doesn't call optimizer.step(), it doesn't benefit from FSDP/DeepSpeed
    # memory sharding. For non-DDP strategies, we keep the model on device without
    # wrapping. SSD already does its own fabric.all_reduce() on importance tensors.
    if distributed_strategy_name != "ddp":
        model = raw_model.to(fabric.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # No fabric.setup - model stays unwrapped, no FSDP communication
    else:
        optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.1)
        model, optimizer = fabric.setup(raw_model, optimizer)  # type: ignore

    criterion = nn.CrossEntropyLoss(reduction="mean")

    if fabric.global_rank == 0 and wandb_logging_flag:
        # Update the config with SSD-specific parameters
        config_dict = {
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "learning_rate": 0.1,
            "dampening_constant": dampening_constant,
            "selection_weighting": selection_weighting,
            "lower_bound": parameters["lower_bound"],
            "exponent": parameters["exponent"],
        }
        wandb.config.update(config_dict)
    fabric.barrier()

    pdr = ParameterPerturber(fabric, model, optimizer, parameters)
    model = model.eval()

    # Calculation of the forget set importances
    forget_importances = pdr.calc_importance(forget_train_dataloader, criterion)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dataloader, criterion)

    # Dampen selected parameters
    pdr.modify_weight(original_importances, forget_importances)

    # Return the post-fabric.setup() wrapped model so unlearn_main.py's
    # gather_full_state_dict() receives a valid FSDP/DeepSpeed wrapper.
    # For DDP/single-GPU this is equivalent to the previous (implicit None)
    # behavior because the caller's `model` already points to the same params.
    return model
