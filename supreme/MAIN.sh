#!/bin/bash
# ==============================================================================
# SUPREME Pipeline Worker: one cell of the experiment grid
# ==============================================================================
# Implements one (training_seed, strategy, dataset, model, forget_target) cell
# of the SUPREME algorithm (training → unlearning → evaluation), with flock-based
# protection against checkpoint race conditions when multiple cells share the
# same training seed.
#
# Loop structure (matches the SUPREME pseudocode):
#   Phase 1 - Training      : Mo = Train(D, s_t)
#   Phase 2 - Unlearning    : for each j: Mr = Train(Dr, s_u);
#                              for each method a: Mu = a(Mo, Df, Dr, s_u)
#   Phase 3 - Evaluation    : for each k: R = Evaluate(Mu, Mr, Df, Dr, E, s_e)
#
# Seed math:
#   s_t = TRAINING_SEED                                  (passed in)
#   s_u = TRAINING_SEED         when J=1                 (paper formula collapse)
#   s_u = TRAINING_SEED*1000+j  when J>1                 (namespaced)
#   s_e = s_u                   when K=1
#   s_e = s_u*1000+k            when K>1
#
# Defaults match the paper run (I=10, J=1, K=1): UNLEARNING_SEEDS=(0),
# EVALUATION_SEEDS=(0). Override via env vars for scaled runs.
#
# Usage (standalone):
#   ./MAIN.sh <training_seed> <strategy> <dataset> <model> <forget_target>
# Usage (SLURM, from run_slurm.sh):
#   Env vars: TRAINING_SEED, STRATEGY, DATASET, MODEL, FORGET_TARGET
#
# Overridable env vars:
#   METHODS            - comma-separated unlearning methods
#   EVAL_METRICS       - comma-separated evaluation metrics
#   UNLEARNING_SEEDS_J - space-separated indices (e.g. "0 1 2 3 4 5 6 7 8 9" for J=10)
#   EVALUATION_SEEDS_K - space-separated indices (e.g. "0 1 2" for K=3)
#   PRECISION          - training/unlearning precision (default 32-true)
#   GPU_ID             - GPU id for standalone runs (default 0)
#   FORCE_RETRAINING   - force retraining even if checkpoint exists
#   FORCE_REUNLEARNING - force re-running unlearning methods
#   FORCE_REEVALUATION - force re-evaluating even if W&B has results
# ==============================================================================

set -e
set -E

export PYTHONUNBUFFERED=1

trap 'exit_code=$?; if [[ $exit_code -ne 0 ]]; then echo ""; echo "ERROR: Command failed at line $LINENO: $BASH_COMMAND"; echo "Exit code: $exit_code"; echo ""; fi' ERR

# ==============================================================================
# Configuration
# ==============================================================================
METHODS="${METHODS:-retrain,original,finetune,bad_teacher,random_labeling,unsir,ssd,lfssd,assd,scrub,jit}"
EVAL_METRICS="${EVAL_METRICS:-accuracy,activation_distance,completeness,jsdiv,layerwise_distance,time,membership_inference_attack}"

# Seed indices (defaults: paper config I=10, J=1, K=1)
IFS=' ' read -r -a UNLEARNING_SEEDS <<< "${UNLEARNING_SEEDS_J:-0}"
IFS=' ' read -r -a EVALUATION_SEEDS <<< "${EVALUATION_SEEDS_K:-0}"
J=${#UNLEARNING_SEEDS[@]}
K=${#EVALUATION_SEEDS[@]}

# Precision
training_precision="${PRECISION:-32-true}"
precision="${PRECISION:-32-true}"

# Execution control
do_training="${DO_TRAINING:-true}"
do_unlearning="${DO_UNLEARNING:-true}"
do_evaluation="${DO_EVALUATION:-true}"
force_retraining="${FORCE_RETRAINING:-false}"
force_reunlearning="${FORCE_REUNLEARNING:-false}"
force_reevaluation="${FORCE_REEVALUATION:-false}"

# Logging
wandb_logging_flag_training=false
wandb_logging_flag_unlearning=false
wandb_logging_flag_evaluation=true
wandb_resume_existing="${WANDB_RESUME_EXISTING:-false}"
export_class_distribution_info_flag=false
track_evaluation_resources=false
cleanup_checkpoints_after_eval="${CLEANUP_CHECKPOINTS_AFTER_EVAL:-false}"

# Training
use_seed_for_training=true
include_gpus_in_path=true

# Advanced
use_nvml_per_process=true
require_nvml_per_process=true
use_process_tracker=false

# Class counts
declare -A N_CLASSES
N_CLASSES["Cifar10"]=10
N_CLASSES["Cifar20"]=20
N_CLASSES["Cifar100"]=100
N_CLASSES["PinsFaceRecognition"]=105
N_CLASSES["Caltech101"]=102

declare -A N_SUPERCLASSES
N_SUPERCLASSES["Cifar20"]=20

declare -A N_SUBCLASSES
N_SUBCLASSES["Cifar20"]=100

# ==============================================================================
# Parse Arguments
# ==============================================================================

if [ -n "$SLURM_JOB_ID" ]; then
    TRAINING_SEED="${TRAINING_SEED:?ERROR: TRAINING_SEED not set}"
    STRATEGY="${STRATEGY:?ERROR: STRATEGY not set}"
    DATASET="${DATASET:?ERROR: DATASET not set}"
    MODEL="${MODEL:?ERROR: MODEL not set}"
    FORGET_TARGET="${FORGET_TARGET:?ERROR: FORGET_TARGET not set}"
else
    TRAINING_SEED="${1:?Usage: $0 <training_seed> <strategy> <dataset> <model> <forget_target>}"
    STRATEGY="${2:?Usage: $0 <training_seed> <strategy> <dataset> <model> <forget_target>}"
    DATASET="${3:?Usage: $0 <training_seed> <strategy> <dataset> <model> <forget_target>}"
    MODEL="${4:?Usage: $0 <training_seed> <strategy> <dataset> <model> <forget_target>}"
    FORGET_TARGET="${5:?Usage: $0 <training_seed> <strategy> <dataset> <model> <forget_target>}"
fi

# Export TRAINING_SEED so unlearn_main.py picks it up for WandB run names
export TRAINING_SEED

# Validate
if [[ ! "$STRATEGY" =~ ^(fullclass|subclass|random_)$ ]]; then
    echo "ERROR: Invalid strategy '$STRATEGY'. Must be fullclass, subclass, or random_."
    exit 1
fi

if [[ ! "$DATASET" =~ ^(Cifar10|Cifar100|Cifar20|PinsFaceRecognition|Caltech101)$ ]]; then
    echo "ERROR: Invalid dataset '$DATASET'."
    exit 1
fi

if [[ ! "$MODEL" =~ ^(ResNet18|ViT)$ ]]; then
    echo "ERROR: Invalid model '$MODEL'. Must be ResNet18 or ViT."
    exit 1
fi

if [ "$STRATEGY" = "subclass" ] && [ "$DATASET" != "Cifar20" ]; then
    echo "ERROR: Subclass strategy only supported for Cifar20."
    exit 1
fi

# ==============================================================================
# Setup
# ==============================================================================

script_dir=$(dirname "$(realpath "$0")")
root_dir=$(realpath "${script_dir}/..")  # script_dir = supreme/ → .. = project root

# Activate the project venv if one isn't already active. Honor $SUPREME_VENV,
# otherwise probe common names (the Makefile default is `unlearning`). Harmless if
# none is found - falls back to whatever python is already on PATH.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    for _venv in "${SUPREME_VENV:-}" unlearning .venv gpu_env venv env; do
        if [ -n "$_venv" ] && [ -f "${root_dir}/${_venv}/bin/activate" ]; then
            source "${root_dir}/${_venv}/bin/activate"
            break
        fi
    done
fi

# GPU setup
if [ -n "$SLURM_JOB_ID" ]; then
    DEVICE_IDS=${CUDA_VISIBLE_DEVICES:-"0"}
    PYTHON_LAUNCHER="srun python"
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_LEVEL=NVL
    export PYTHONFAULTHANDLER=1
    NUM_GPUS=${SLURM_NTASKS:-1}
else
    DEVICE_IDS="${GPU_ID:-0}"
    PYTHON_LAUNCHER="CUDA_VISIBLE_DEVICES=$DEVICE_IDS python"
    IFS=',' read -r -a gpu_array <<< "$DEVICE_IDS"
    NUM_GPUS=${#gpu_array[@]}
fi

gpu_path_component=""
[ "$include_gpus_in_path" = true ] && gpu_path_component="${NUM_GPUS}gpus/"

# Distributed-strategy path segment - must match train_main.py:290 and
# update_checkpoint_paths.py:93 so the TRAINING_DONE marker is written to
# the same timestamp dir that find_checkpoint later inspects.
if [ "$NUM_GPUS" -gt 1 ]; then
    if [ "${DISTRIBUTED_STRATEGY:-ddp}" = "deepspeed" ]; then
        dist_str_component="dist_deepspeed_stage${DEEPSPEED_STAGE:-2}/"
    else
        dist_str_component="dist_${DISTRIBUTED_STRATEGY:-ddp}/"
    fi
else
    dist_str_component="no_dist/"
fi

IFS=',' read -r -a method_array <<< "$METHODS"

# Export environment
[ "$use_nvml_per_process" = true ] && export USE_NVML_PER_PROCESS=1 || unset USE_NVML_PER_PROCESS
[ "$require_nvml_per_process" = true ] && export REQUIRE_NVML_PER_PROCESS=1 || unset REQUIRE_NVML_PER_PROCESS
export LOG_PER_PROCESS_DATA="false"
export USE_FABRIC_CALLBACKS="false"
export WANDB_LOG_EVALUATION="$wandb_logging_flag_evaluation"

WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-R32}"
export WANDB_PROJECT_PREFIX

echo "=============================================="
echo "SUPREME Pipeline - Cell Worker"
echo "=============================================="
echo "Training Seed: $TRAINING_SEED"
echo "Strategy: $STRATEGY"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Forget Target: $FORGET_TARGET"
echo "Methods: $METHODS"
echo "Eval Metrics: $EVAL_METRICS"
echo "Unlearning Seeds (J=$J): ${UNLEARNING_SEEDS[*]}"
echo "Evaluation Seeds (K=$K): ${EVALUATION_SEEDS[*]}"
echo "Precision: $precision"
echo "WandB Prefix: $WANDB_PROJECT_PREFIX"
echo "Num GPUs: $NUM_GPUS"
echo "=============================================="
echo ""

# ==============================================================================
# Helper Functions
# ==============================================================================

find_checkpoint() {
    local net_type=$1
    local dataset=$2
    python "${root_dir}/supreme/utils/unlearning/update_checkpoint_paths.py" \
        -precision "$training_precision" \
        -training_seed "$TRAINING_SEED" \
        -unlearning_seed "$TRAINING_SEED" \
        -num_gpus "$NUM_GPUS" \
        -include_gpus_in_path "$include_gpus_in_path" \
        -net "$net_type" \
        -dataset "$dataset"
}

train_model() {
    local model_type=$1
    local dataset=$2
    local num_classes=$3
    local purpose=${4:-"N/A"}

    echo "=============================================="
    echo "TRAINING: $model_type on $dataset ($num_classes classes)"
    echo "Training Precision: $training_precision | Training Seed: $TRAINING_SEED"
    echo "=============================================="

    local train_cmd="${PYTHON_LAUNCHER} ${root_dir}/supreme/utils/training/train_main.py \
        -unlearning_seed \"$TRAINING_SEED\" \
        -precision \"$training_precision\" \
        -net \"$model_type\" \
        -dataset \"$dataset\" \
        -classes \"$num_classes\" \
        -unlearning_context \"$purpose\" \
        -include_gpus_in_path \"$include_gpus_in_path\""

    [ "$use_seed_for_training" = true ] && train_cmd+=" -training_seed \"$TRAINING_SEED\""
    [ "$wandb_logging_flag_training" = true ] && train_cmd+=" -wandb_logging_flag"

    eval "$train_cmd"
}

run_unlearning_method() {
    local method=$1
    local extra_args=$2
    local net_type=$3
    local dataset=$4
    local type_of_unlearning_strategy=$5
    local full_weight_path=$6

    if [ "$type_of_unlearning_strategy" = "random_" ] && [ "$method" = "unsir" ]; then
        echo "SKIPPING: UNSIR is not compatible with random_ strategy"
        return 0
    fi

    # Skip if checkpoint exists (unless force_reunlearning=true)
    if [ "$force_reunlearning" = false ] && [ -n "$LOG_DIR" ]; then
        local method_capitalized="${method^}"
        local existing_method_dir="$LOG_DIR/$method_capitalized"
        local existing_model_path="$existing_method_dir/${method_capitalized}_model.pth"
        local existing_time_path="$existing_method_dir/${method_capitalized}_time_elapsed.json"
        local existing_memory_path="$existing_method_dir/${method_capitalized}_memory_usage.json"
        local existing_power_path="$existing_method_dir/${method_capitalized}_compute_utilisation.json"

        if [[ -e "$existing_model_path" && -e "$existing_time_path" && -e "$existing_memory_path" && -e "$existing_power_path" ]]; then
            echo "SKIP UNLEARNING: '$method' checkpoint exists at $existing_method_dir"
            [ "$do_evaluation" = true ] && run_evaluation_loop "$method" "$extra_args" "$net_type" "$dataset" "$type_of_unlearning_strategy" "$full_weight_path"
            return 0
        fi
    fi

    echo "=============================================="
    echo "UNLEARNING: $method on $dataset ($net_type)"
    echo "Precision: $precision | Strategy: $type_of_unlearning_strategy"
    echo "Training Seed: $TRAINING_SEED | Unlearning Seed: $current_unlearning_seed"
    echo "=============================================="

    local forget_script="${root_dir}/supreme/utils/unlearning/unlearn_main.py"
    # PERFORM_EVALUATION=false here → unlearn_main.py only runs the unlearning stage, so -seed = s_u
    local base_cmd="\"$forget_script\" -precision \"$precision\" -net \"$net_type\" -dataset \"$dataset\" -type_of_unlearning_strategy \"$type_of_unlearning_strategy\" -weight_path \"$full_weight_path\" -seed \"$current_unlearning_seed\" -eval_metrics \"${EVAL_METRICS}\" $extra_args"

    [ "$wandb_logging_flag_unlearning" = true ] && base_cmd+=" -wandb_logging_flag"
    [ "$use_process_tracker" = true ] && base_cmd+=" -use_process_tracker"
    [ "$force_reunlearning" = true ] && base_cmd+=" -force_reunlearning"

    local cmd="${PYTHON_LAUNCHER} ${base_cmd}"

    eval "$cmd -method $method"

    [ "$do_evaluation" = true ] && run_evaluation_loop "$method" "$extra_args" "$net_type" "$dataset" "$type_of_unlearning_strategy" "$full_weight_path"
}

run_evaluation() {
    local method=$1
    local extra_args=$2
    local net_type=$3
    local dataset=$4
    local type_of_unlearning_strategy=$5
    local full_weight_path=$6

    echo ""
    echo "----------------------------------------------"
    echo "EVALUATING: $method on $dataset ($net_type)"
    echo "Training Seed: $TRAINING_SEED | Unlearning Seed: $current_unlearning_seed | Evaluation Seed: $current_evaluation_seed"
    echo "----------------------------------------------"

    # Save and unset SLURM variables for single-GPU evaluation
    local original_cuda_visible_devices=$CUDA_VISIBLE_DEVICES
    local original_slurm_ntasks=$SLURM_NTASKS
    local original_slurm_procid=$SLURM_PROCID
    local original_slurm_localid=$SLURM_LOCALID
    local original_slurm_job_id=$SLURM_JOB_ID

    if [ -n "$SLURM_JOB_ID" ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
        unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_JOB_ID
        unset SLURM_NTASKS_PER_NODE SLURM_NNODES SLURM_NODEID
    else
        export CUDA_VISIBLE_DEVICES=${gpu_array[0]:-0}
    fi

    local eval_script="${root_dir}/supreme/utils/unlearning/unlearn_main.py"
    # PERFORM_EVALUATION=true here → unlearn_main.py only runs the evaluation stage, so -seed = s_e
    local eval_cmd="python \"$eval_script\" -method \"$method\" -eval_metrics \"${EVAL_METRICS}\" -net \"$net_type\" -dataset \"$dataset\" -type_of_unlearning_strategy \"$type_of_unlearning_strategy\" -seed \"$current_evaluation_seed\" -precision \"$precision\" -weight_path \"$full_weight_path\" $extra_args"

    [ "$wandb_logging_flag_evaluation" = true ] && eval_cmd+=" -wandb_logging_flag"
    [ "$cleanup_checkpoints_after_eval" = true ] && eval_cmd+=" -cleanup_checkpoints_after_eval"

    # Check WandB for existing results
    if [ "$force_reevaluation" = false ]; then
        local check_script="${root_dir}/supreme/utils/wandb_utils/runtime/wandb_setup.py"
        local check_cmd="python \"$check_script\" -method \"$method\" -net \"$net_type\" -dataset \"$dataset\" -type_of_unlearning_strategy \"$type_of_unlearning_strategy\" -seed \"$current_evaluation_seed\" -precision \"$precision\" -eval_metrics \"${EVAL_METRICS}\" $extra_args"

        if eval "$check_cmd"; then
            echo "SKIPPING: Results already exist in WandB for $method (train_seed=$TRAINING_SEED, unlearn_seed=$current_unlearning_seed, eval_seed=$current_evaluation_seed)"
            export CUDA_VISIBLE_DEVICES="$original_cuda_visible_devices"
            [ -n "$original_slurm_job_id" ] && export SLURM_JOB_ID="$original_slurm_job_id"
            [ -n "$original_slurm_ntasks" ] && export SLURM_NTASKS="$original_slurm_ntasks"
            [ -n "$original_slurm_procid" ] && export SLURM_PROCID="$original_slurm_procid"
            [ -n "$original_slurm_localid" ] && export SLURM_LOCALID="$original_slurm_localid"
            return 0
        fi
    fi

    export PERFORM_EVALUATION=true
    [ "$wandb_resume_existing" = true ] && export WANDB_RESUME_EXISTING=true
    eval "$eval_cmd"
    unset PERFORM_EVALUATION
    unset WANDB_RESUME_EXISTING

    export CUDA_VISIBLE_DEVICES="$original_cuda_visible_devices"
    [ -n "$original_slurm_job_id" ] && export SLURM_JOB_ID="$original_slurm_job_id"
    [ -n "$original_slurm_ntasks" ] && export SLURM_NTASKS="$original_slurm_ntasks"
    [ -n "$original_slurm_procid" ] && export SLURM_PROCID="$original_slurm_procid"
    [ -n "$original_slurm_localid" ] && export SLURM_LOCALID="$original_slurm_localid"
    return 0
}

# K-loop wrapper: evaluate the same Mu/Mr against each evaluation seed
run_evaluation_loop() {
    local saved_eval_seed="$current_evaluation_seed"
    for k in "${EVALUATION_SEEDS[@]}"; do
        if [ "$K" -eq 1 ]; then
            current_evaluation_seed="$current_unlearning_seed"
        else
            current_evaluation_seed=$((current_unlearning_seed * 1000 + k))
        fi
        run_evaluation "$@"
    done
    current_evaluation_seed="$saved_eval_seed"
}

# ==============================================================================
# Phase 1: Training (locked)
# ==============================================================================

echo "=============================================="
echo "PHASE 1: TRAINING (seed=$TRAINING_SEED, model=$MODEL)"
echo "=============================================="

n_classes=${N_CLASSES["$DATASET"]}
if [ "$DATASET" = "Cifar20" ]; then
    training_purpose="fullclass_subclass"
elif [ "$DATASET" = "Cifar10" ]; then
    training_purpose="random"
else
    training_purpose="fullclass"
fi
if [ -z "$n_classes" ]; then
    echo "ERROR: No class count defined for dataset '$DATASET'."
    exit 1
fi

# Flock prevents race conditions when multiple cells share the same (TRAINING_SEED, MODEL, DATASET):
# the marker file is only written after train_model returns successfully, so any task that sees it
# is guaranteed a fully-trained checkpoint. Always-on - cheap when J=1 (single cell per cluster).
LOCK_DIR="${root_dir}/logs/training/.locks"
mkdir -p "$LOCK_DIR"
LOCK_FILE="${LOCK_DIR}/train_seed_${TRAINING_SEED}_${MODEL}_${DATASET}.lock"
TRAINING_DONE_MARKER="${LOCK_DIR}/train_seed_${TRAINING_SEED}_${MODEL}_${DATASET}.done"

# Training body, factored so it can run under either lock implementation below.
# Robust by design: it does NOT rely on `set -e` (bash suppresses `set -e` inside
# the `||`-guarded mkdir-lock path below, and historically that let a failed train
# fall through and write the "done" marker with no checkpoint). It marks training
# complete only when the trainer exits 0 AND a real *-best.pth checkpoint exists.
_train_if_needed() {
    if [ -f "$TRAINING_DONE_MARKER" ] && [ "$force_retraining" = false ]; then
        local existing_checkpoint
        existing_checkpoint=$(find_checkpoint "$MODEL" "$DATASET" 2>/dev/null || true)
        echo "Training already complete for $MODEL/$DATASET (seed=$TRAINING_SEED). Skipping."
        echo "  Path: $existing_checkpoint"
        return 0
    fi

    echo "Training $MODEL on $DATASET with seed=$TRAINING_SEED..."
    if ! train_model "$MODEL" "$DATASET" "$n_classes" "$training_purpose"; then
        echo "ERROR: training failed for $MODEL/$DATASET (seed=$TRAINING_SEED); not marking complete."
        return 1
    fi

    # Confirm the trainer actually produced a checkpoint before marking done -
    # otherwise a future run would skip training yet find no checkpoint (deadlock).
    local base_ckpt_dir ts_dir final_ckpt
    base_ckpt_dir="${root_dir}/logs/training/precision_${precision}/${gpu_path_component}${dist_str_component}train_seed_${TRAINING_SEED}/unlearning_seed_${TRAINING_SEED}/model_checkpoints/${MODEL}/${DATASET}"
    ts_dir=$(ls -td "${base_ckpt_dir}"/*/ 2>/dev/null | head -1)
    final_ckpt=""
    [ -n "$ts_dir" ] && final_ckpt=$(ls -t "${ts_dir}"*-best.pth 2>/dev/null | head -1)
    if [ -z "$final_ckpt" ]; then
        echo "ERROR: training reported success but no *-best.pth was produced under"
        echo "       $base_ckpt_dir - not marking complete."
        return 1
    fi

    # Only now is the run guaranteed a usable checkpoint: stamp both markers.
    printf '%s\n' "$final_ckpt" > "${ts_dir}TRAINING_DONE"
    echo "Wrote TRAINING_DONE marker → ${ts_dir}TRAINING_DONE"
    touch "$TRAINING_DONE_MARKER"
}

if [ "$do_training" = true ]; then
    # Serialize trainers of the same (seed, model, dataset). Real flock(1) on
    # Linux is released automatically when the subshell's fd 200 closes. On hosts
    # without flock - e.g. macOS - fall back to a portable mkdir spinlock (mkdir
    # is atomic), released explicitly so a failed train cannot deadlock later runs.
    if command -v flock >/dev/null 2>&1; then
        (
            flock -x 200
            _train_if_needed
        ) 200>"$LOCK_FILE"
    else
        _lockdir="${LOCK_FILE}.d"
        while ! mkdir "$_lockdir" 2>/dev/null; do sleep 0.2; done
        _train_rc=0
        ( _train_if_needed ) || _train_rc=$?
        rmdir "$_lockdir" 2>/dev/null || true
        [ "$_train_rc" -eq 0 ] || exit "$_train_rc"
    fi
fi

weight_path=$(find_checkpoint "$MODEL" "$DATASET" 2>/dev/null || true)
if [ -z "$weight_path" ]; then
    echo "ERROR: No checkpoint found for $MODEL/$DATASET after training phase."
    echo "  Expected at: logs/training/precision_${precision}/${gpu_path_component}${dist_str_component}train_seed_${TRAINING_SEED}/unlearning_seed_${TRAINING_SEED}/model_checkpoints/$MODEL/$DATASET/"
    exit 1
fi
echo "Using checkpoint: $weight_path"
echo ""

# ==============================================================================
# Phase 2 & 3: Unlearning + Evaluation
# ==============================================================================

echo "=============================================="
echo "PHASE 2 + 3: UNLEARNING + EVALUATION"
echo "Strategy: $STRATEGY | Dataset: $DATASET | Model: $MODEL"
echo "Forget Target: $FORGET_TARGET"
echo "Training Seed: $TRAINING_SEED | J=$J unlearning seed(s) | K=$K evaluation seed(s)"
echo "=============================================="
echo ""

for j in "${UNLEARNING_SEEDS[@]}"; do
    # Seed math: collapse to TRAINING_SEED for J=1 (matches paper formula s_u = s_t when J=1);
    # namespace by *1000 for J>1 to keep unlearning seeds globally unique across training seeds.
    if [ "$J" -eq 1 ]; then
        current_unlearning_seed="$TRAINING_SEED"
    else
        current_unlearning_seed=$((TRAINING_SEED * 1000 + j))
    fi
    current_evaluation_seed="$current_unlearning_seed"  # initial value; K-loop overrides

    echo ""
    echo "========================================================"
    echo "Unlearning Seed: $current_unlearning_seed (j=$j, Training Seed: $TRAINING_SEED)"
    echo "========================================================"

    # The seed variable used by unlearn_main.py
    seed="$current_unlearning_seed"

    # Export s_u so unlearn_main.py / wandb_setup.py can detect K>1 eval runs
    # (where the -seed CLI arg holds s_e ≠ s_u) and emit the `_eseed{E}` suffix.
    # During the unlearning phase UNLEARNING_SEED == -seed, so the old
    # `tseed{T}_useed{U}` form is preserved (no eseed component added).
    export UNLEARNING_SEED="$current_unlearning_seed"

    echo ""
    echo "--- Processing: $STRATEGY target '$FORGET_TARGET' (unlearn_seed=$current_unlearning_seed) ---"

    # Build log directory with two-level seed structure
    case "$STRATEGY" in
        "fullclass")
            main_dir="${root_dir}/logs/unlearning/precision_${precision}/${gpu_path_component}train_seed_${TRAINING_SEED}/unlearn_seed_${current_unlearning_seed}/fullclass/${DATASET}/${MODEL}/classes_${n_classes}"
            unlearning_args="-classes $n_classes -forget_class_name $FORGET_TARGET"
            sub_dir="${main_dir}/forget_class_${FORGET_TARGET}"
            ;;
        "subclass")
            local_n_superclasses=${N_SUPERCLASSES["$DATASET"]}
            local_n_subclasses=${N_SUBCLASSES["$DATASET"]}
            main_dir="${root_dir}/logs/unlearning/precision_${precision}/${gpu_path_component}train_seed_${TRAINING_SEED}/unlearn_seed_${current_unlearning_seed}/subclass/${DATASET}/${MODEL}/superclasses_${local_n_superclasses}_subclasses_${local_n_subclasses}"
            unlearning_args="-superclasses $local_n_superclasses -subclasses $local_n_subclasses -forget_subclass_name $FORGET_TARGET"
            sub_dir="${main_dir}/forget_class_${FORGET_TARGET}"
            ;;
        "random_")
            main_dir="${root_dir}/logs/unlearning/precision_${precision}/${gpu_path_component}train_seed_${TRAINING_SEED}/unlearn_seed_${current_unlearning_seed}/random_/${DATASET}/${MODEL}/classes_${n_classes}"
            unlearning_args="-classes $n_classes -forget_perc $FORGET_TARGET"
            sub_dir="${main_dir}/forget_perc_${FORGET_TARGET}"
            ;;
    esac
    mkdir -p "$sub_dir"
    export LOG_DIR="$sub_dir"

    # Mr retraining lifted out of the method loop: Mr depends only on (Dr, s_u, forget_target),
    # not on the unlearning method a. Producing it once per (training_seed, unlearning_seed,
    # forget_target) matches the pseudocode and avoids |A| identical retrains. The existence
    # check inside run_unlearning_method preserves crash-recovery behavior.
    needs_mr=false
    if [[ $(python3 -c "import sys; sys.path.append('${root_dir}'); from supreme.utils.unlearning.unlearn_main import requires_retrain; print(requires_retrain('${EVAL_METRICS}'.split(',')))" 2>/dev/null) == "True" ]]; then
        needs_mr=true
    fi
    for m in "${method_array[@]}"; do
        [ "$m" = "retrain" ] && needs_mr=true && break
    done
    if [ "$needs_mr" = true ] && { [ "$do_unlearning" = true ] || [ "$do_evaluation" = true ]; }; then
        run_unlearning_method "retrain" "$unlearning_args" "$MODEL" "$DATASET" "$STRATEGY" "$weight_path"
    fi

    for method in "${method_array[@]}"; do
        [ "$method" = "retrain" ] && continue  # Mr handled above, outside the method loop
        if [ "$do_unlearning" = true ] || [ "$do_evaluation" = true ]; then
            run_unlearning_method "$method" "$unlearning_args" "$MODEL" "$DATASET" "$STRATEGY" "$weight_path"
        fi
    done

    echo "All methods completed for unlearning seed $current_unlearning_seed"
done

echo ""
echo "=============================================="
echo "CELL COMPLETED"
echo "Training Seed: $TRAINING_SEED | Strategy: $STRATEGY | Dataset: $DATASET"
echo "Model: $MODEL | Forget Target: $FORGET_TARGET"
echo "=============================================="
