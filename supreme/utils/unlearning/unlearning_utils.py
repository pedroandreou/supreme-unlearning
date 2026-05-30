import torch
from torch.utils.data import Dataset, ConcatDataset, random_split, Subset
from supreme.utils.training.training_utils import (
    get_lr,
    training_step,
    evaluate,
    prepare_dataloaders,
)
from supreme.utils.debug_utils import export_retain_forget_class_distribution
import os
import wandb
from supreme.utils.generic_utils import create_dataloader, get_root_directory
from supreme.utils import project_config
import supreme.datasets.datasets as dataset_module


class _ListDataset(Dataset):
    """A simple dataset that wraps a list of data points."""

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_classwise_indices(ds, num_labels, type_of_unlearning_strategy=None):
    """Build a mapping from class ID to list of dataset indices.

    Instead of storing full (img, label, clabel) tuples, we only store the
    indices into the original dataset. This reduces memory usage and file
    sizes dramatically (from ~10GB to ~100KB for typical datasets).

    OPTIMIZATION: For fullclass strategy, use dataset.targets directly if available
    (ImageFolder-based datasets). This avoids loading images just to read labels,
    making it ~100x faster for large datasets.

    Args:
        ds: The dataset to classify
        num_labels: Number of classes
        type_of_unlearning_strategy: "fullclass" groups by clabel, "subclass" groups by label

    Returns:
        dict mapping class_id -> list of indices into ds
    """
    classwise_indices = {i: [] for i in range(num_labels)}

    if type_of_unlearning_strategy == "fullclass":
        # OPTIMIZATION: Use targets directly if available (avoids loading images)
        # ImageFolder-based datasets (PinsFaceRecognition, Caltech101, etc.) have targets
        # IMPORTANT: For Cifar20, use coarse_targets instead of targets (fine labels)
        if hasattr(ds, 'coarse_targets'):
            # Fast path for Cifar20: use coarse labels
            for idx, clabel in enumerate(ds.coarse_targets):
                if clabel not in classwise_indices:
                    classwise_indices[clabel] = []
                classwise_indices[clabel].append(idx)
        elif hasattr(ds, 'targets'):
            # Fast path: read labels directly without loading images
            for idx, clabel in enumerate(ds.targets):
                if clabel not in classwise_indices:
                    classwise_indices[clabel] = []
                classwise_indices[clabel].append(idx)
        else:
            # Slow path: load each sample to get label (fallback for custom datasets)
            for idx in range(len(ds)):
                _, _, clabel = ds[idx]
                if clabel not in classwise_indices:
                    classwise_indices[clabel] = []
                classwise_indices[clabel].append(idx)
    else:  # subclass
        # For subclass, we need the fine-grained label (position 1)
        # This requires loading samples as there's no direct attribute
        for idx in range(len(ds)):
            _, label, _ = ds[idx]
            if label not in classwise_indices:
                classwise_indices[label] = []
            classwise_indices[label].append(idx)

    return classwise_indices


def prepare_classwise_dataloaders(
    fabric,
    num_gpus,
    model_name,
    dataset_name,
    num_labels,
    batch_size,
    type_of_unlearning_strategy,
    **kwargs,
):
    # Creates datasets for method execution
    def build_retain_forget_indices(
        classwise_train_indices, classwise_test_indices, num_labels, forget_class_id
    ):
        """Build lists of indices for retain and forget sets.

        Args:
            classwise_train_indices: dict mapping class_id -> list of train indices
            classwise_test_indices: dict mapping class_id -> list of test indices
            num_labels: Number of classes
            forget_class_id: The class to forget

        Returns:
            Tuple of (retain_train_indices, retain_test_indices,
                     forget_train_indices, forget_test_indices)
        """
        forget_train_indices, retain_train_indices = [], []
        forget_test_indices, retain_test_indices = [], []

        for cls in range(num_labels):
            if cls == forget_class_id:
                forget_train_indices.extend(classwise_train_indices[cls])
                forget_test_indices.extend(classwise_test_indices[cls])
            else:
                retain_train_indices.extend(classwise_train_indices[cls])
                retain_test_indices.extend(classwise_test_indices[cls])

        return (retain_train_indices, retain_test_indices, forget_train_indices, forget_test_indices)

    seed = kwargs.get("seed", None)
    forget_class_name = kwargs.get("forget_class_name", None)
    assert all(
        x is not None for x in [seed, forget_class_name]
    ), "One or more required arguments (seed, forget_class_name) is None"

    export_class_distribution_info_flag = kwargs.get(
        "export_class_distribution_info_flag", False
    )

    final_output_dir = os.getenv("LOG_DIR")
    assert final_output_dir is not None, "LOG_DIR environment variable must be set"

    # Use centralized path construction for consistency with prepare_dataloaders()
    dataset_path = project_config.get_dataset_path_from_log_dir(final_output_dir, model_name)

    # Paths for original dataset files
    trainset_full_path = os.path.join(dataset_path, "trainset.pt")
    testset_full_path = os.path.join(dataset_path, "testset.pt")
    fabric.print(f"trainset_full_path: {trainset_full_path}")
    fabric.print(f"testset_full_path: {testset_full_path}")

    # Index-based paths (much smaller files ~100KB vs ~10GB)
    retain_train_indices_path = os.path.join(final_output_dir, "retain_train_indices.pt")
    retain_test_indices_path = os.path.join(final_output_dir, "retain_test_indices.pt")
    forget_train_indices_path = os.path.join(final_output_dir, "forget_train_indices.pt")
    forget_test_indices_path = os.path.join(final_output_dir, "forget_test_indices.pt")
    fabric.print(f"retain_train_indices_path: {retain_train_indices_path}")
    fabric.print(f"retain_test_indices_path: {retain_test_indices_path}")
    fabric.print(f"forget_train_indices_path: {forget_train_indices_path}")
    fabric.print(f"forget_test_indices_path: {forget_test_indices_path}")

    ###########################################################################
    # Full tensor paths - fallback for legacy experiments that saved the full
    # retain/forget datasets as .pt files (multi-GB) instead of indices (~100KB).
    # This fallback will be used when index files are not found but the old
    # full tensor files exist on disk.
    ###########################################################################
    retain_train_tensor_path = os.path.join(final_output_dir, "retain_train.pt")
    retain_test_tensor_path = os.path.join(final_output_dir, "retain_test.pt")
    forget_train_tensor_path = os.path.join(final_output_dir, "forget_train.pt")
    forget_test_tensor_path = os.path.join(final_output_dir, "forget_test.pt")

    # Load datasets if they already exist
    retain_train = None
    retain_test = None

    forget_train = None
    forget_test = None

    trainset = None
    testset = None
    train_dataloader = None
    test_dataloader = None

    index_paths = [
        trainset_full_path,
        testset_full_path,
        retain_train_indices_path,
        retain_test_indices_path,
        forget_train_indices_path,
        forget_test_indices_path,
    ]

    full_tensor_paths = [
        trainset_full_path,
        testset_full_path,
        retain_train_tensor_path,
        retain_test_tensor_path,
        forget_train_tensor_path,
        forget_test_tensor_path,
    ]

    if all(os.path.exists(path) for path in index_paths):
        fabric.print(
            "Loading index-based datasets (fast loading, ~100KB files)..."
        )

        if fabric.global_rank == 0:
            # Load original datasets
            trainset = torch.load(trainset_full_path, weights_only=False)
            fabric.print("Loaded train set")
            testset = torch.load(testset_full_path, weights_only=False)
            fabric.print("Loaded test set")

            # Load indices (tiny files, instant load)
            retain_train_indices = torch.load(retain_train_indices_path, weights_only=False)
            fabric.print(f"Loaded retain train indices ({len(retain_train_indices)} samples)")
            retain_test_indices = torch.load(retain_test_indices_path, weights_only=False)
            fabric.print(f"Loaded retain test indices ({len(retain_test_indices)} samples)")
            forget_train_indices = torch.load(forget_train_indices_path, weights_only=False)
            fabric.print(f"Loaded forget train indices ({len(forget_train_indices)} samples)")
            forget_test_indices = torch.load(forget_test_indices_path, weights_only=False)
            fabric.print(f"Loaded forget test indices ({len(forget_test_indices)} samples)")
        else:
            retain_train_indices = None
            retain_test_indices = None
            forget_train_indices = None
            forget_test_indices = None
        fabric.barrier()

        trainset = fabric.broadcast(trainset, src=0)
        fabric.print("Broadcasted train set")
        testset = fabric.broadcast(testset, src=0)
        fabric.print("Broadcasted test set")
        retain_train_indices = fabric.broadcast(retain_train_indices, src=0)
        retain_test_indices = fabric.broadcast(retain_test_indices, src=0)
        forget_train_indices = fabric.broadcast(forget_train_indices, src=0)
        forget_test_indices = fabric.broadcast(forget_test_indices, src=0)
        fabric.print("Broadcasted all indices")

        # Reconstruct Subset views from indices
        retain_train = Subset(trainset, retain_train_indices)
        forget_train = Subset(trainset, forget_train_indices)

        # For random_ strategy:
        #   - retain_test = entire testset (verify model still works on all test data)
        #   - forget_test = same samples as forget_train (no natural test set for random samples)
        # For fullclass/subclass:
        #   - retain_test = test samples of retained classes
        #   - forget_test = test samples of forgotten class
        retain_test = Subset(testset, retain_test_indices)
        if type_of_unlearning_strategy == "random_":
            forget_test = Subset(trainset, forget_test_indices)  # Uses trainset, same as forget_train
        else:
            forget_test = Subset(testset, forget_test_indices)  # Uses testset
        fabric.print("Reconstructed Subset views from indices")

        train_dataloader = create_dataloader(
            dataset=trainset,
            batch_size=batch_size,
            is_training=True,
            num_gpus=num_gpus,
        )
        fabric.print("Created train loader")

        test_dataloader = create_dataloader(
            dataset=testset,
            batch_size=batch_size,
            is_training=False,
            num_gpus=num_gpus,
        )
        fabric.print("Created test loader")

    ###########################################################################
    # FALLBACK: Full tensor .pt files (legacy code)
    # If index files are not found but old full tensor files exist, load them
    # directly. These are the multi-GB retain_train.pt, retain_test.pt,
    # forget_train.pt, forget_test.pt files from the previous implementation.
    #
    # WARNING: This fallback is DEPRECATED. Users should transition to the
    # index-based format which is ~100x smaller and faster to load.
    # To transition, simply delete the old full tensor files:
    #   rm retain_train.pt retain_test.pt forget_train.pt forget_test.pt
    # and re-run the experiment. The new index files (*_indices.pt) will be
    # generated automatically and saved for future runs.
    ###########################################################################
    elif all(os.path.exists(path) for path in full_tensor_paths):
        fabric.print(
            "###########################################################################"
        )
        fabric.print(
            "# WARNING: Loading LEGACY full tensor .pt files (retain_train.pt, etc.)   #"
        )
        fabric.print(
            "# These files are large (multi-GB) and slow to load.                      #"
        )
        fabric.print(
            "# To transition to the new index-based format (~100KB, instant load):      #"
        )
        fabric.print(
            "#   1. Delete the old files: rm retain_train.pt retain_test.pt             #"
        )
        fabric.print(
            "#      forget_train.pt forget_test.pt                                     #"
        )
        fabric.print(
            "#   2. Re-run the experiment. New *_indices.pt files will be generated.    #"
        )
        fabric.print(
            "###########################################################################"
        )

        if fabric.global_rank == 0:
            trainset = torch.load(trainset_full_path, weights_only=False)
            fabric.print("Loaded train set")
            testset = torch.load(testset_full_path, weights_only=False)
            fabric.print("Loaded test set")

            retain_train = torch.load(retain_train_tensor_path, weights_only=False)
            fabric.print(f"Loaded retain train set ({len(retain_train)} samples)")
            retain_test = torch.load(retain_test_tensor_path, weights_only=False)
            fabric.print(f"Loaded retain test set ({len(retain_test)} samples)")
            forget_train = torch.load(forget_train_tensor_path, weights_only=False)
            fabric.print(f"Loaded forget train set ({len(forget_train)} samples)")
            forget_test = torch.load(forget_test_tensor_path, weights_only=False)
            fabric.print(f"Loaded forget test set ({len(forget_test)} samples)")
        fabric.barrier()

        trainset = fabric.broadcast(trainset, src=0)
        fabric.print("Broadcasted train set")
        testset = fabric.broadcast(testset, src=0)
        fabric.print("Broadcasted test set")
        retain_train = fabric.broadcast(retain_train, src=0)
        fabric.print("Broadcasted retain train set")
        retain_test = fabric.broadcast(retain_test, src=0)
        fabric.print("Broadcasted retain test set")
        forget_train = fabric.broadcast(forget_train, src=0)
        fabric.print("Broadcasted forget train set")
        forget_test = fabric.broadcast(forget_test, src=0)
        fabric.print("Broadcasted forget test set")

        # Wrap lists in _ListDataset so they work as Dataset objects
        retain_train = _ListDataset(retain_train)
        retain_test = _ListDataset(retain_test)
        forget_train = _ListDataset(forget_train)
        forget_test = _ListDataset(forget_test)

        train_dataloader = create_dataloader(
            dataset=trainset,
            batch_size=batch_size,
            is_training=True,
            num_gpus=num_gpus,
        )
        fabric.print("Created train loader")

        test_dataloader = create_dataloader(
            dataset=testset,
            batch_size=batch_size,
            is_training=False,
            num_gpus=num_gpus,
        )
        fabric.print("Created test loader")
    ###########################################################################
    # END FALLBACK
    ###########################################################################

    else:
        precision = kwargs.get("precision", None)
        assert precision is not None, "precision must be provided for unlearning"

        trainset, testset, train_dataloader, test_dataloader = prepare_dataloaders(
            fabric=fabric,
            num_gpus=num_gpus,
            precision=precision,
            model_name=model_name,
            seed=seed,
            dataset_name=dataset_name,
            type_of_unlearning_strategy=type_of_unlearning_strategy,
            batch_size=batch_size,
            unlearning=True,
            export_class_distribution_info_flag=export_class_distribution_info_flag,
        )

        if type_of_unlearning_strategy == "random_":
            forget_perc = kwargs.get("forget_perc", None)
            assert (
                forget_perc is not None
            ), "forget_perc must be provided for random_ unlearning"

            # Split the training set into forget and retain subsets
            # random_split returns Subset objects with .indices attribute
            forget_train_subset, retain_train_subset = random_split(
                trainset, [forget_perc, 1 - forget_perc]
            )

            # Extract indices from Subsets
            forget_train_indices = list(forget_train_subset.indices)
            retain_train_indices = list(retain_train_subset.indices)
            # For random_, forget_test = forget_train (same train indices, not test!)
            # and retain_test = full test set
            forget_test_indices = forget_train_indices  # Uses trainset, not testset
            retain_test_indices = list(range(len(testset)))

        else:  # fullclass and subclass
            classwise_train_indices = get_classwise_indices(
                trainset, num_labels, type_of_unlearning_strategy
            )
            classwise_test_indices = get_classwise_indices(
                testset, num_labels, type_of_unlearning_strategy
            )

            fabric.print("The classwise train has classes: ", len(classwise_train_indices))
            fabric.print("The classwise test has classes: ", len(classwise_test_indices))

            forget_class_id = kwargs.get("forget_class_id", None)
            assert forget_class_id is not None, "forget_class_id cannot be None"
            (
                retain_train_indices,
                retain_test_indices,
                forget_train_indices,
                forget_test_indices,
            ) = build_retain_forget_indices(
                classwise_train_indices, classwise_test_indices, num_labels, forget_class_id
            )

        fabric.print("the train set is of size: ", len(trainset))
        fabric.print("the test set is of size: ", len(testset))

        if fabric.global_rank == 0:
            # Save indices (tiny files ~100KB instead of ~10GB)
            fabric.print(f"Saving retain train indices to {retain_train_indices_path}")
            torch.save(retain_train_indices, retain_train_indices_path)
            fabric.print(f"the retain train set is of size: {len(retain_train_indices)}")

            fabric.print(f"Saving retain test indices to {retain_test_indices_path}")
            torch.save(retain_test_indices, retain_test_indices_path)
            fabric.print(f"the retain test set is of size: {len(retain_test_indices)}")

            fabric.print(f"Saving forget train indices to {forget_train_indices_path}")
            torch.save(forget_train_indices, forget_train_indices_path)
            fabric.print(f"the forget train set is of size: {len(forget_train_indices)}")

            fabric.print(f"Saving forget test indices to {forget_test_indices_path}")
            torch.save(forget_test_indices, forget_test_indices_path)
            fabric.print(f"the forget test set is of size: {len(forget_test_indices)}")
        fabric.barrier()

        # Create Subset views from indices
        retain_train = Subset(trainset, retain_train_indices)
        retain_test = Subset(testset, retain_test_indices)
        forget_train = Subset(trainset, forget_train_indices)
        # For random_ strategy, forget_test uses trainset (same as forget_train)
        # For fullclass/subclass, forget_test uses testset
        if type_of_unlearning_strategy == "random_":
            forget_test = Subset(trainset, forget_test_indices)
        else:
            forget_test = Subset(testset, forget_test_indices)
        fabric.print("Created Subset views from indices")

    if os.getenv("SAMPLE_SCALING") == "true":
        # === TEMPORARY: DUPLICATE DATASETS FOR EXPERIMENTS ===
        # To create a larger unlearning set for distributed speedup experiments, the factor is set from run_local.sh
        duplication_factor = int(os.getenv("DUPLICATION_FACTOR", "1"))
        fabric.print(f"[EXPERIMENT] Duplication factor from env: {duplication_factor}")

        # # Duplicate the training sets for unlearning
        # retain_train = ConcatDataset([retain_train] * duplication_factor)
        # fabric.print(f"[EXPERIMENT] Duplicated retain_train size: {len(retain_train)} samples")

        # forget_train = ConcatDataset([forget_train] * duplication_factor)
        # fabric.print(f"[EXPERIMENT] Duplicated forget_train size: {len(forget_train)} samples")

        # Duplicate the test sets for unlearning (using ConcatDataset for Subset compatibility)
        testset = ConcatDataset([testset] * duplication_factor)
        fabric.print(f"[EXPERIMENT] Duplicated testset size: {len(testset)} samples")
        retain_test = ConcatDataset([retain_test] * duplication_factor)
        fabric.print(
            f"[EXPERIMENT] Duplicated retain_test size: {len(retain_test)} samples"
        )
        forget_test = ConcatDataset([forget_test] * duplication_factor)
        fabric.print(
            f"[EXPERIMENT] Duplicated forget_test size: {len(forget_test)} samples"
        )
        # === END TEMPORARY BLOCK ===

    datasets = [
        trainset,
        testset,
        retain_train,
        retain_test,
        forget_train,
        forget_test,
    ]
    if any(dataset is None for dataset in datasets):
        raise ValueError(
            "One of the datasets is None. Check the paths and dataset generation logic."
        )

    # Subset is already a Dataset, so we can pass it directly to create_dataloader
    # (no need for _ListDataset wrapper)
    forget_train_dataloader = create_dataloader(
        dataset=forget_train,
        batch_size=batch_size,
        # THIS IS AN INCOSISTENCY BETWEEN BAD TEACHER AND SSD REPOS, WE FOLLOWED BAD TEACHER'S REPO
        # SINCE IT IS PART OF THE TRAINING DATASET, THEN SHOULD BE SHUFFLED, SSD REPO DOES NOT SHUFFLE IT
        is_training=True,
        num_gpus=num_gpus,
    )
    retain_train_dataloader = create_dataloader(
        dataset=retain_train,
        batch_size=batch_size,
        is_training=True,
        num_gpus=num_gpus,
    )
    fabric.print("Created retain train dataloader")

    forget_test_dataloader = create_dataloader(
        dataset=forget_test,
        batch_size=batch_size,
        is_training=False,
        num_gpus=num_gpus,
    )
    fabric.print("Created forget test dataloader")
    retain_test_dataloader = create_dataloader(
        dataset=retain_test,
        batch_size=batch_size,
        is_training=False,
        num_gpus=num_gpus,
    )
    fabric.print("Created retain test dataloader")

    full_train_dataloader = create_dataloader(
        dataset=ConcatDataset([retain_train, forget_train]),
        batch_size=batch_size,
        is_training=True,
        num_gpus=num_gpus,
    )
    fabric.print("Created full train dataloader")

    if fabric.global_rank == 0 and export_class_distribution_info_flag:
        export_retain_forget_class_distribution(
            fabric=fabric,
            num_gpus=num_gpus,
            seed=seed,
            batch_size=batch_size,
            dataset_name=dataset_name,
            type_of_unlearning_strategy=type_of_unlearning_strategy,
            original_classes=trainset.classes,
            forget_train=forget_train,
            retain_train=retain_train,
            forget_test=forget_test,
            retain_test=retain_test,
            forget_class_name=forget_class_name,
        )
    fabric.barrier()

    # For retrain: create a retain dataloader with training augmentation.
    # The standard retain_train_dataloader uses unlearning transforms (no augmentation),
    # but retrain needs the same augmentation pipeline as pretraining for a fair baseline.
    retain_train_augmented_dataloader = None
    if hasattr(retain_train, "indices"):
        img_size = 224 if model_name == "ViT" else 32
        root = get_root_directory(dataset_name)
        augmented_trainset = getattr(dataset_module, dataset_name)(
            root=root, download=False, train=True, unlearning=False, img_size=img_size,
            model_name=model_name,
        )
        augmented_retain_train = Subset(augmented_trainset, retain_train.indices)
        retain_train_augmented_dataloader = create_dataloader(
            dataset=augmented_retain_train,
            batch_size=batch_size,
            is_training=True,
            num_gpus=num_gpus,
        )
        fabric.print(
            f"Created augmented retain train dataloader for retrain "
            f"({len(augmented_retain_train)} samples with training transforms)"
        )

    return {
        "retain_train_dataloader": retain_train_dataloader,
        "retain_test_dataloader": retain_test_dataloader,
        "forget_train_dataloader": forget_train_dataloader,
        "forget_test_dataloader": forget_test_dataloader,
        "trainset": trainset,
        "testset": testset,
        "train_dataloader": train_dataloader,
        "test_dataloader": test_dataloader,
        "full_train_dataloader": full_train_dataloader,
        "retain_train_augmented_dataloader": retain_train_augmented_dataloader,
    }


def fit_one_unlearning_cycle(
    fabric,
    num_gpus,
    epochs,
    model,
    train_dataloader,
    test_dataloader,
    lr,
    wandb_logging_flag,
):
    history = []

    raw_model = model.module if hasattr(model, "module") else model
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr)

    model, optimizer = fabric.setup(raw_model, optimizer)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    if fabric.global_rank == 0 and wandb_logging_flag:
        # Update the config
        config_dict = {
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "learning_rate": lr,
        }
        wandb.config.update(config_dict)
    fabric.barrier()

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []

        # if fabric.global_rank == 0:
        #     fabric.call("on_train_epoch_start", fabric=fabric, epoch=epoch)
        # fabric.barrier()

        for batch_idx, batch in enumerate(train_dataloader):
            # if fabric.global_rank == 0:
            #     fabric.call("on_train_batch_start")
            # fabric.barrier()

            loss = training_step(model=model, batch=batch, criterion=criterion)
            fabric.backward(loss)
            train_losses.append(loss.detach().cpu())

            optimizer.step()
            optimizer.zero_grad()

            current_lr = get_lr(optimizer)
            lrs.append(current_lr)

            # if fabric.global_rank == 0:
            #     fabric.call(
            #         "on_train_batch_end",
            #         loss=loss,
            #         epoch=epoch,
            #         batch_idx=batch_idx,
            #         lr=current_lr,
            #     )
            # fabric.barrier()

        # Calculate average train loss for the epoch
        train_losses = torch.stack(train_losses).mean()

        # if fabric.global_rank == 0:
        #     fabric.call(
        #         "on_train_epoch_end",
        #         epoch=epoch,
        #         train_loss=train_losses,
        #         last_lr=lrs[-1],
        #     )
        # fabric.barrier()

        # # Testing phase
        # if fabric.global_rank == 0:
        #     fabric.call("on_test_epoch_start", fabric=fabric)
        # fabric.barrier()

        result_dict = evaluate(
            fabric=fabric,
            model=model,
            test_dataloader=test_dataloader,
            epoch=epoch,
            do_global_aggregation=False,
        )
        result = result_dict["metric_value_dict"]

        # if fabric.global_rank == 0:
        #     # Call test end
        #     fabric.call(
        #         "on_test_epoch_end",
        #         epoch=epoch,
        #         loss=result["Loss"]["final_value"],
        #         acc=result["Acc"]["final_value"],
        #     )
        # fabric.barrier()

        # Store complete results in history
        result["train_loss"] = train_losses
        result["lrs"] = lrs
        history.append(result)

    # Return BOTH the wrapped model AND history (see fit_one_learning_cycle
    # for the rationale - FSDP requires callers to use the post-setup wrapper).
    return model, history
