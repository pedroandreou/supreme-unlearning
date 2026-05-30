import os
import csv
from torch.utils.data import Dataset
import torch
import supreme.utils.project_config as project_config
import time
import traceback
from supreme.utils.generic_utils import create_dataloader
import debugpy
import socket
from contextlib import contextmanager
import logging
import sys

output_dir_name = os.path.join(project_config.PROJECT_ROOT, "logs", "dataset_distributions")


############################################################################
########### FOR PRINTING DISTRIBUTIONS OF TRAINING AND TEST SETS ###########
############################################################################
class CustomDataset(Dataset):
    def __init__(self, data, classes):
        self.data = data
        self.classes = classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # Return (img, label, clabel)


def create_output_directories(
    fabric,
    output_dir,
    seed,
    type_of_unlearning_strategy,
    dataset_name,
    forget_class_name=None,
    set_type=None,
    last_dir_name=None,
):
    components = [output_dir]

    if seed is not None:
        components.append(f"seed_{seed}")

    if type_of_unlearning_strategy is not None:
        components.append(type_of_unlearning_strategy)

    components.append(
        dataset_name
    )  # This will add "Cifar10" or whatever dataset was chosen

    for component in [forget_class_name, set_type, last_dir_name]:
        if component is not None:
            components.append(component)

    final_output_dir = os.path.join(*components)

    if fabric.global_rank == 0:
        os.makedirs(final_output_dir, exist_ok=True)
    fabric.barrier()

    return final_output_dir


def export_retain_forget_class_distribution(
    fabric,
    num_gpus,
    seed,
    batch_size,
    dataset_name,
    type_of_unlearning_strategy,
    original_classes,
    forget_train,
    retain_train,
    forget_test,
    retain_test,
    forget_class_name=None,
):
    # Export forget training set
    forget_train_dataset = CustomDataset(forget_train, original_classes)

    forget_train_dl = create_dataloader(
        dataset=forget_train_dataset,
        batch_size=batch_size,
        is_training=True,
        num_gpus=num_gpus,
    )
    export_train_test_data(
        fabric=fabric,
        seed=seed,
        dataset_name=dataset_name,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        loader=forget_train_dl,
        set_type="train",
        subset_type="forget",
        forget_class_name=forget_class_name,
    )

    # Export retain training set
    retain_train_dataset = CustomDataset(retain_train, original_classes)

    retain_train_dl = create_dataloader(
        dataset=retain_train_dataset,
        batch_size=batch_size,
        is_training=True,
        num_gpus=num_gpus,
    )
    export_train_test_data(
        fabric=fabric,
        seed=seed,
        dataset_name=dataset_name,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        loader=retain_train_dl,
        set_type="train",
        subset_type="retain",
        forget_class_name=forget_class_name,
    )

    # Export forget test set
    forget_test_dataset = CustomDataset(forget_test, original_classes)

    forget_test_dl = create_dataloader(
        dataset=forget_test_dataset,
        batch_size=batch_size,
        is_training=False,
        num_gpus=num_gpus,
    )
    export_train_test_data(
        fabric=fabric,
        seed=seed,
        dataset_name=dataset_name,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        loader=forget_test_dl,
        set_type="test",
        subset_type="forget",
        forget_class_name=forget_class_name,
    )

    # Export retain test set
    retain_test_dataset = CustomDataset(retain_test, original_classes)

    retain_test_dl = create_dataloader(
        dataset=retain_test_dataset,
        batch_size=batch_size,
        is_training=False,
        num_gpus=num_gpus,
    )
    export_train_test_data(
        fabric=fabric,
        seed=seed,
        dataset_name=dataset_name,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        loader=retain_test_dl,
        set_type="test",
        subset_type="retain",
        forget_class_name=forget_class_name,
    )


def export_train_test_data(
    fabric,
    seed,
    dataset_name,
    type_of_unlearning_strategy,
    loader,
    set_type,
    forget_class_name=None,
    subset_type=None,
):
    # Call both functions with the same arguments
    export_dataset_info(
        fabric=fabric,
        seed=seed,
        dataset_name=dataset_name,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        forget_class_name=forget_class_name,
        dataloader=loader,
        set_type=set_type,
        subset_type=subset_type,
    )
    export_class_distribution(
        fabric=fabric,
        seed=seed,
        dataset_name=dataset_name,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        forget_class_name=forget_class_name,
        dataloader=loader,
        set_type=set_type,
        subset_type=subset_type,
    )


def get_class_name(dataloader, label):
    dataset = dataloader.dataset
    while hasattr(dataset, "dataset"):  # Handle nested datasets (e.g., Subset)
        dataset = dataset.dataset

    return dataset.classes[label.item()]


def get_superclass_name(class_id):
    """Get the name of the superclass based on its ID"""
    return project_config.cifar20_dict_inverted.get(class_id, "Unknown Class")


def export_dataset_info(
    fabric,
    seed,
    dataset_name,
    type_of_unlearning_strategy,
    forget_class_name,
    dataloader,
    set_type,
    subset_type=None,
):
    if fabric.global_rank == 0:
        os.makedirs(output_dir_name, exist_ok=True)
    fabric.barrier()

    final_output_dir = create_output_directories(
        fabric=fabric,
        output_dir=output_dir_name,
        seed=seed,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        dataset_name=dataset_name,
        forget_class_name=forget_class_name,
        set_type=set_type,
        last_dir_name="set_info",
    )

    # Create full file path using final_output_dir
    set_info_name = f"{subset_type}_set_info.csv" if subset_type else "set_info.csv"

    # If subset_type is None (meaning it's "set_info.csv"), go up three directories
    if subset_type is None:
        # Go three directories back
        final_output_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(final_output_dir))
        )

    full_path = os.path.join(final_output_dir, set_info_name)

    # Check if file already exists
    if os.path.exists(full_path):
        # print(f"File {full_path} already exists. Skipping writing.")
        return  # Exit the function early if the file already exists

    # If the file doesn't exist, proceed with writing the data
    if any(dataset_name == x for x in ["Cifar100", "Cifar10", "PinsFaceRecognition"]):
        with open(full_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Batch", "Index", "Class"])

            for batch_idx, batch in enumerate(dataloader):
                labels = batch[-1]  # The last item in the batch is the labels
                for idx, label in enumerate(labels):
                    class_name = get_class_name(dataloader, label)
                    writer.writerow([batch_idx, idx, class_name])

    elif dataset_name == "Cifar20":
        with open(full_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Batch", "Index", "Superclass", "Subclass"])

            for batch_idx, batch in enumerate(dataloader):
                subclasses = batch[-2]  # subclasses are the second-last item
                superclasses = batch[-1]  # superclasses are the last item

                # Convert to class names
                subclass_names = [
                    get_class_name(dataloader, subclass) for subclass in subclasses
                ]
                superclass_ids = [superclass.item() for superclass in superclasses]
                superclass_names = [
                    get_superclass_name(class_id) for class_id in superclass_ids
                ]

                for idx, (superclass_name, subclass_name) in enumerate(
                    zip(superclass_names, subclass_names)
                ):
                    # Write in the order: batch_idx, index, superclass, subclass
                    writer.writerow([batch_idx, idx, superclass_name, subclass_name])

    else:  # When trying to find out about the structure/columns of a new dataset when it comes in and I am not sure about
        with open(full_path, "w", newline="") as file:
            writer = csv.writer(file)

            # Dynamically write header based on the number of items in each batch
            first_batch = next(iter(dataloader))
            header = ["Batch", "Index"] + [f"Item_{i}" for i in range(len(first_batch))]
            writer.writerow(header)

            # Loop through the batches
            for batch_idx, batch in enumerate(dataloader):
                for idx in range(
                    len(batch[0])
                ):  # Assuming all elements in the batch have the same length
                    row = [batch_idx, idx]  # Start with batch number and index
                    row.extend(
                        [
                            str(batch[i][idx].tolist())  # type: ignore
                            if isinstance(batch[i][idx], torch.Tensor)  # type: ignore
                            else batch[i][idx]  # type: ignore
                            for i in range(len(batch))
                        ]
                    )
                    writer.writerow(row)


def export_class_distribution(
    fabric,
    seed,
    dataset_name,
    type_of_unlearning_strategy,
    forget_class_name,
    dataloader,
    set_type,
    subset_type=None,
):
    if fabric.global_rank == 0:
        os.makedirs(output_dir_name, exist_ok=True)
    fabric.barrier()

    final_output_dir = create_output_directories(
        fabric=fabric,
        output_dir=output_dir_name,
        seed=seed,
        type_of_unlearning_strategy=type_of_unlearning_strategy,
        dataset_name=dataset_name,
        forget_class_name=forget_class_name,
        set_type=set_type,
        last_dir_name="class_distribution",
    )

    # Create full file path using final_output_dir
    class_distribution_name = (
        f"{subset_type}_class_distribution.csv"
        if subset_type
        else "class_distribution.csv"
    )
    full_path = os.path.join(final_output_dir, class_distribution_name)

    def get_class_distribution(dataset_name, type_of_unlearning_strategy, dataloader):
        if type_of_unlearning_strategy == "subclass":
            subclass_counts = {}
            superclass_counts = {}

            for batch_idx, batch in enumerate(dataloader):
                if len(batch) < 3:  # Ensure there are enough elements in the batch
                    print("Unexpected batch structure:", type(batch))
                    continue

                subclasses = batch[-2]  # Assuming subclasses are second last
                superclasses = batch[-1]  # Assuming superclasses are last

                if not (
                    isinstance(subclasses, torch.Tensor)
                    and isinstance(superclasses, torch.Tensor)
                ):
                    print(
                        "Expected subclasses and superclasses to be Tensors, got:",
                        type(subclasses),
                        type(superclasses),
                    )
                    continue

                # Count subclass occurrences
                for subclass in subclasses:
                    subclass_name = get_class_name(dataloader, subclass)
                    subclass_counts[subclass_name] = (
                        subclass_counts.get(subclass_name, 0) + 1
                    )

                # Count superclass occurrences
                for superclass in superclasses:
                    superclass_id = superclass.item()
                    superclass_name = get_superclass_name(superclass_id)
                    superclass_counts[superclass_name] = (
                        superclass_counts.get(superclass_name, 0) + 1
                    )

            return subclass_counts, superclass_counts

        else:
            class_counts = {}

            for batch_idx, batch in enumerate(dataloader):
                if len(batch) < 3:  # Ensure there are enough elements in the batch
                    print("Unexpected batch structure:", type(batch))
                    continue

                classes = batch[-1]

                if not (isinstance(classes, torch.Tensor)):
                    print(
                        "Expected subclasses and superclasses to be Tensors, got:",
                        type(classes),
                    )
                    continue

                # Count class occurrences
                if dataset_name == "Cifar20":
                    for class_ in classes:
                        class_id = class_.item()
                        class_name = get_superclass_name(class_id)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                else:
                    for class_ in classes:
                        class_name = get_class_name(dataloader, class_)
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

            return class_counts

    # Get the counts based on the unlearning strategy
    if type_of_unlearning_strategy == "subclass":
        subclass_counts, superclass_counts = get_class_distribution(
            dataset_name, type_of_unlearning_strategy, dataloader
        )

        # Write subclass and superclass counts to CSV
        with open(full_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Class Type", "Class Name", "Count"])

            # Write subclass counts if any
            if subclass_counts:
                for subclass_name, count in subclass_counts.items():
                    writer.writerow(["Subclass", subclass_name, count])

            # Write superclass counts if any
            if superclass_counts:
                for superclass_name, count in superclass_counts.items():
                    writer.writerow(["Superclass", superclass_name, count])

    else:  # fullclass or random_
        class_counts = get_class_distribution(
            dataset_name, type_of_unlearning_strategy, dataloader
        )

        # Write subclass and superclass counts to CSV
        with open(full_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Class Type", "Class Name", "Count"])

            # Write class counts if any
            if class_counts:
                for class_name, count in class_counts.items():  # type: ignore
                    writer.writerow(["Class", class_name, count])


############################################################################
############################################################################
############################################################################


def benchmark_dataloader(fabric, num_gpus, dataset, batch_size):
    """
    It is suggested that num_workers should be set to the number of CPUs for faster performance
    See https://pytorch-lightning.readthedocs.io/en/0.10.0/performance.html
    """

    def time_loader_iteration(dataset, batch_size, n_workers):
        """Benchmark a dataloader with specific number of workers and find the time taken"""

        loader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            is_training=True,
            num_gpus=num_gpus,
        )

        start = time.time()
        # Iterate through a small subset of the data for benchmarking
        for i, _ in enumerate(loader):
            if i >= 20:  # Only test with 20 batches
                break
        end = time.time()
        return end - start

    fabric.print("\nBenchmarking different num_workers configurations...")
    worker_options = [0, 1, 2, 4, 8, 16]
    benchmark_results = {}
    for n_workers in worker_options:
        try:
            time_taken = time_loader_iteration(dataset, batch_size, n_workers)
            benchmark_results[n_workers] = time_taken
            fabric.print(f"Workers: {n_workers}, Time: {time_taken:.2f}s")
        except Exception as e:
            fabric.print(f"Error with {n_workers} workers: {str(e)}")
    # Find optimal number of workers (minimum time)
    optimal_workers = min(benchmark_results.items(), key=lambda x: x[1])[0]
    fabric.print(f"\nOptimal number of workers: {optimal_workers}")


############################################################################
############################################################################
############################################################################


class DebuggerPort:
    def __init__(self, port=5678):
        self.port = port
        self.sock = None

    def is_port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", self.port)) == 0

    def cleanup(self):
        try:
            if self.sock:
                self.sock.close()
                self.sock = None  # Prevent multiple cleanups
        except Exception as e:
            logging.error(f"Error during debugger cleanup: {e}")


@contextmanager
def create_debugger_session(port=5678):
    debugger = DebuggerPort(port)
    if not debugger.is_port_in_use():
        try:
            debugpy.listen(("0.0.0.0", port))
            print("⚡ VS Code debugger can now be attached ⚡")
            debugpy.wait_for_client()
            print("🎯 VS Code debugger attached!")
        except Exception as e:
            logging.error(f"Failed to start debugger: {e}")
    try:
        yield
    finally:
        debugger.cleanup()


def handle_distributed_error(fabric, error, epoch=None):
    """Handles errors in distributed training, coordinating debugging through rank 0"""
    # All ranks report their errors
    print(f"\n{'='*50}")
    print(f"**Error** occurred on rank {fabric.local_rank}")

    # Add detailed error information
    error_details = {
        "Error Type": type(error).__name__,
        "Error Message": str(error),
        "Location": "Unknown",  # Will be populated from traceback
    }

    # Get detailed traceback information
    tb = traceback.extract_tb(sys.exc_info()[2])

    # Find the most relevant frame (usually the last non-framework call)
    relevant_frames = []
    for frame in tb:
        if not any(x in frame.filename for x in ["lightning", "torch", "fabric"]):
            relevant_frames.append(
                {
                    "file": frame.filename,
                    "line": frame.lineno,
                    "function": frame.name,
                    "code": frame.line,
                }
            )

    if relevant_frames:
        last_frame = relevant_frames[-1]
        error_details["Location"] = (
            f"{last_frame['file']}:{last_frame['line']} in {last_frame['function']}"
        )
        error_details["Code Context"] = last_frame["code"]

    # Print detailed error information
    print("\nDetailed Error Information:")
    for key, value in error_details.items():
        print(f"{key}: {value}")

    print("\nFull Traceback:")
    traceback.print_exc()

    # Add model initialization specific debugging if relevant
    if "Model not initialized" in str(error):
        print("\nModel Initialization Debug Info:")
        print("- Checking for transformers library...")
        try:
            import transformers

            print(f"  √ transformers version: {transformers.__version__}")
        except ImportError:
            print("  × transformers library not found!")

        print("- Checking GPU availability...")
        try:
            import torch

            if torch.cuda.is_available():
                print(f"  √ CUDA available: True")
                print(f"  √ CUDA version: {torch.version.cuda}")
            elif torch.backends.mps.is_available():
                print(f"  √ MPS (Apple Silicon) available: True")
            else:
                print(f"  × No GPU available (CUDA: False, MPS: False)")
        except Exception as e:
            print(f"  × GPU check failed: {e}")

    try:
        # Synchronize all processes and collect error info
        error_info = {
            "rank": fabric.local_rank,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "details": error_details,
        }
        all_errors = fabric.all_gather(error_info)

        # Only rank 0 coordinates debugging
        if fabric.global_rank == 0:
            print("\nCollected errors from all processes:")
            for err in all_errors:
                print(f"\nRank {err['rank']}:")
                print(f"Error: {err['error']}")
                print("Details:")
                for k, v in err["details"].items():
                    print(f"  {k}: {v}")
                print("Traceback:")
                print(err["traceback"])

        fabric.barrier()

    except Exception as gather_error:
        print(
            f"Error during error handling on rank {fabric.local_rank}: {gather_error}"
        )

    print(f"{'='*50}\n")
    # Re-raise the original error to ensure proper process termination
    raise error
