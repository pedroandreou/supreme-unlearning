import torch
from supreme.utils.unlearning.evaluation_utils import track_evaluation_metric

# Paper: "Towards Making Systems Forget with Machine Unlearning" at https://ieeexplore.ieee.org/document/7163042
# and implementation can be found at https://github.com/theLauA/MachineUnlearningPy/blob/b59dcd1d6d028b7807a56897b3911fd6989f0c02/lenskit/algorithms/item_knn.py#L141
# but the implementation was not very clear
# so we tried to make our own implementations based on the description of the below papers


"""
Paper which introduced this metric initially was: "Towards Making Systems Forget with Machine Unlearning"
at https://www.ieee-security.org/TC/SP2015/papers-archived/6949a463.pdf
and the authors explicitly defined Completeness as the percentage of input samples where the models produce identical predictions
"""


@track_evaluation_metric
def calculate_completeness(
    fabric, model1, model2, test_dataloader, do_global_aggregation=True
):
    """
    Calculate completeness between an unlearned model and a model trained from scratch.
    Completeness is calculated as both the percentage of input samples with identical predictions
    from both models and the Jaccard distance between the two models' predictions.

    Returns:
    - completeness_percentage: Percentage of identical predictions
    """
    # # Track epoch start
    # calculate_completeness.track_epoch_start(fabric, 0, "completeness")

    total_samples = 0
    identical_predictions = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # calculate_completeness.track_batch_start(fabric)

            x, _, y = batch

            # Get predictions from both models
            outputs_model1 = model1(x)
            outputs_model2 = model2(x)

            # Convert outputs to predicted classes and move to CPU
            _, pred_model1 = torch.max(outputs_model1, 1)
            _, pred_model2 = torch.max(outputs_model2, 1)

            # Move tensors to CPU before comparison
            pred_model1_cpu = pred_model1.detach().cpu()
            pred_model2_cpu = pred_model2.detach().cpu()

            # Perform comparison on CPU
            batch_identical = torch.sum(pred_model1_cpu == pred_model2_cpu).item()
            identical_predictions += batch_identical
            total_samples += pred_model1.size(0)

            # # Track batch end with current batch completeness
            # batch_completeness = (batch_identical / pred_model1.size(0)) * 100
            # calculate_completeness.track_batch_end(
            #     fabric, batch_idx, 0, batch_completeness
            # )

    # Gather from all processes
    if do_global_aggregation:
        gathered_all_identical_predictions = fabric.all_gather(identical_predictions)
        gathered_all_total_samples = fabric.all_gather(total_samples)
        total_identical_predictions = gathered_all_identical_predictions.sum().item()
        total_samples = gathered_all_total_samples.sum().item()
    else:
        total_identical_predictions = identical_predictions
        # total_samples is already the correct local value
        gathered_all_identical_predictions = torch.tensor([total_identical_predictions])
        gathered_all_total_samples = torch.tensor([total_samples])

    completeness_percentage = (
        (total_identical_predictions / total_samples) * 100 if total_samples > 0 else 0
    )

    # # Track epoch end with final completeness percentage
    # calculate_completeness.track_epoch_end(fabric, 0, completeness_percentage)

    completeness_dict = {
        "final_value": completeness_percentage,
        "per_process_identical_predictions": gathered_all_identical_predictions.tolist(),
        "per_process_total_samples": gathered_all_total_samples.tolist(),
    }

    # Return a dictionary
    return completeness_dict
