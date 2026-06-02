#!/bin/bash
# ==============================================================================
# SUPREME Pipeline - Local Dispatcher
# ==============================================================================
# Iterates the configured experiment grid (seed × strategy × dataset × model ×
# forget_target) and invokes supreme/MAIN.sh per cell sequentially.
#
# Usage:
#   ./run_local.sh --gpu <gpu_ids> [--strategies <s>] [--datasets <d>] [--models <m>]
#                  [--methods <m>] [--metrics <m>] [--training-seeds <s>] [--forget-percs <p>]
#                  [--fullclass-classes <c>] [--subclass-classes <c>] [--force-retraining]
#                  [--unlearning-seeds "<j ...>"] [--evaluation-seeds "<k ...>"]
#
# Seed-protocol flags (default: matched, J=K=1):
#   --training-seeds   "260,261,...,269"  outer loop, I = |S_T| training seeds
#   --unlearning-seeds "0"                matched protocol (s_u = s_t)
#   --unlearning-seeds "0 1 ... 9"        decoupled protocol with J=10 (s_u = s_t * 1000 + j)
#   --evaluation-seeds "0"                K=1 (s_e = s_u)
#   --evaluation-seeds "0 1 2"            K=3 (s_e = s_u * 1000 + k)
#
# Examples:
#   ./run_local.sh --gpu 0
#   ./run_local.sh --gpu 0 --training-seeds 260 --strategies random_ --datasets Cifar10 \
#                  --models ResNet18 --methods retrain --forget-percs 0.01
#
# For SLURM submission, use supreme/run_slurm.sh instead.
# ==============================================================================

set -e
set -E
export PYTHONUNBUFFERED=1

trap 'exit_code=$?; if [[ $exit_code -ne 0 ]]; then echo ""; echo "ERROR: Command failed at line $LINENO: $BASH_COMMAND"; echo "Exit code: $exit_code"; echo ""; fi' ERR

# ==============================================================================
# Defaults (overridable via CLI flags)
# ==============================================================================
DEFAULT_STRATEGIES="fullclass,subclass,random_"
DEFAULT_DATASETS="Cifar10,Cifar20,Cifar100,PinsFaceRecognition,Caltech101"
DEFAULT_MODELS="ResNet18,ViT"
DEFAULT_METHODS="retrain,original,finetune,bad_teacher,random_labeling,unsir,ssd,lfssd,assd,scrub,jit"
DEFAULT_EVAL_METRICS="accuracy,activation_distance,completeness,jsdiv,layerwise_distance,time,membership_inference_attack"
DEFAULT_SEEDS="260,261,262,263,264,265,266,267,268,269"
DEFAULT_FORGET_PERCS="0.001,0.005,0.01,0.05,0.10"

# Forget targets (per dataset, for fullclass/subclass strategies)
FULLCLASS_CIFAR20="vehicle2 veg people electrical_devices natural_scenes"
FULLCLASS_CIFAR100="rocket mushroom baby lamp sea"
FULLCLASS_PINSFACE="1 10 20 30 40"
FULLCLASS_CALTECH101="airplanes car_side chair elephant lamp"
SUBCLASS_CIFAR20="rocket mushroom baby lamp sea"

# Per-cell defaults forwarded to MAIN.sh
PRECISION="${PRECISION:-32-true}"
FORCE_RETRAINING="${FORCE_RETRAINING:-false}"
WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-R32}"
# Seed-protocol indices: space-separated, default "0" (J=1, K=1 - matched protocol)
UNLEARNING_SEEDS_J="${UNLEARNING_SEEDS_J:-0}"
EVALUATION_SEEDS_K="${EVALUATION_SEEDS_K:-0}"

# ==============================================================================
# Parse CLI flags
# ==============================================================================
DEVICE_IDS="0"
CLI_STRATEGIES=""
CLI_DATASETS=""
CLI_MODELS=""
CLI_METHODS=""
CLI_METRICS=""
CLI_SEEDS=""
CLI_FORGET_PERCS=""
CLI_FULLCLASS_CLASSES=""
CLI_SUBCLASS_CLASSES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-g)                DEVICE_IDS="$2";              shift 2 ;;
        --strategies|-s)         CLI_STRATEGIES="$2";          shift 2 ;;
        --datasets|-d)           CLI_DATASETS="$2";            shift 2 ;;
        --models|-m)             CLI_MODELS="$2";              shift 2 ;;
        --methods)               CLI_METHODS="$2";             shift 2 ;;
        --metrics)               CLI_METRICS="$2";             shift 2 ;;
        --training-seeds)        CLI_SEEDS="$2";               shift 2 ;;
        --forget-percs)          CLI_FORGET_PERCS="$2";        shift 2 ;;
        --fullclass-classes)     CLI_FULLCLASS_CLASSES="$2";   shift 2 ;;
        --subclass-classes)      CLI_SUBCLASS_CLASSES="$2";    shift 2 ;;
        --force-retraining)      FORCE_RETRAINING="true";      shift ;;
        --precision)             PRECISION="$2";               shift 2 ;;
        --wandb-prefix)          WANDB_PROJECT_PREFIX="$2";    shift 2 ;;
        --unlearning-seeds)      UNLEARNING_SEEDS_J="$2";      shift 2 ;;
        --evaluation-seeds)      EVALUATION_SEEDS_K="$2";      shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_local.sh --gpu <ids> [--strategies <s>] [--datasets <d>] [--models <m>] [--methods <m>] [--metrics <m>] [--training-seeds <s>] [--forget-percs <p>] [--unlearning-seeds <\"0 1 2\">] [--evaluation-seeds <\"0\">] [--force-retraining] [--precision <p>]"
            exit 1 ;;
    esac
done

UNLEARNING_STRATEGIES="${CLI_STRATEGIES:-$DEFAULT_STRATEGIES}"
DATASETS="${CLI_DATASETS:-$DEFAULT_DATASETS}"
MODELS="${CLI_MODELS:-$DEFAULT_MODELS}"
METHODS_LIST="${CLI_METHODS:-$DEFAULT_METHODS}"
EVAL_METRICS_LIST="${CLI_METRICS:-$DEFAULT_EVAL_METRICS}"
SEEDS_ARG="${CLI_SEEDS:-$DEFAULT_SEEDS}"
FORGET_PERCS_ARG="${CLI_FORGET_PERCS:-$DEFAULT_FORGET_PERCS}"

IFS=',' read -ra SEED_VALUES <<< "$SEEDS_ARG"
IFS=',' read -ra STRATEGY_VALUES <<< "$UNLEARNING_STRATEGIES"
IFS=',' read -ra DATASET_VALUES <<< "$DATASETS"
IFS=',' read -ra MODEL_VALUES <<< "$MODELS"
IFS=',' read -ra FORGET_PERC_VALUES <<< "$FORGET_PERCS_ARG"

# Per-(dataset, model) fullclass / subclass forget targets
declare -A FULLCLASS_TARGETS
FULLCLASS_TARGETS["Cifar20"]="${CLI_FULLCLASS_CLASSES:-$FULLCLASS_CIFAR20}"
FULLCLASS_TARGETS["Cifar100"]="${CLI_FULLCLASS_CLASSES:-$FULLCLASS_CIFAR100}"
FULLCLASS_TARGETS["PinsFaceRecognition"]="${CLI_FULLCLASS_CLASSES:-$FULLCLASS_PINSFACE}"
FULLCLASS_TARGETS["Caltech101"]="${CLI_FULLCLASS_CLASSES:-$FULLCLASS_CALTECH101}"

declare -A SUBCLASS_TARGETS
SUBCLASS_TARGETS["Cifar20"]="${CLI_SUBCLASS_CLASSES:-$SUBCLASS_CIFAR20}"

# ==============================================================================
# GPU setup (passed through to MAIN.sh via GPU_ID)
# ==============================================================================
export GPU_ID="$DEVICE_IDS"

echo "=============================================="
echo "SUPREME Pipeline - Local Dispatcher"
echo "=============================================="
echo "Strategies: $UNLEARNING_STRATEGIES"
echo "Datasets: $DATASETS"
echo "Models: $MODELS"
echo "Methods: $METHODS_LIST"
echo "Eval Metrics: $EVAL_METRICS_LIST"
echo "Training seeds: ${SEED_VALUES[*]}"
echo "Forget percentages (random_): ${FORGET_PERC_VALUES[*]}"
echo "GPUs: $DEVICE_IDS"
echo "Precision: $PRECISION"
echo "Force retraining: $FORCE_RETRAINING"
echo "WandB prefix: $WANDB_PROJECT_PREFIX"
echo "Unlearning seed indices j (J=$(echo $UNLEARNING_SEEDS_J | wc -w | tr -d ' ')): $UNLEARNING_SEEDS_J  ->  seed = training_seed (J=1), else training_seed*1000+j"
echo "Evaluation seed indices  k (K=$(echo $EVALUATION_SEEDS_K | wc -w | tr -d ' ')): $EVALUATION_SEEDS_K  ->  seed = unlearning_seed (K=1), else unlearning_seed*1000+k"
echo "=============================================="
echo ""

# Forward shared config to MAIN.sh via env
export METHODS="$METHODS_LIST"
export EVAL_METRICS="$EVAL_METRICS_LIST"
export PRECISION
export FORCE_RETRAINING
export WANDB_PROJECT_PREFIX
export UNLEARNING_SEEDS_J
export EVALUATION_SEEDS_K

# ==============================================================================
# Per-(dataset, strategy) target enumeration
# ==============================================================================
enumerate_targets() {
    local strategy=$1
    local dataset=$2

    case "$strategy" in
        "fullclass")
            local targets="${FULLCLASS_TARGETS[$dataset]:-}"
            [ -z "$targets" ] && return 1
            echo "$targets"
            ;;
        "subclass")
            local targets="${SUBCLASS_TARGETS[$dataset]:-}"
            [ -z "$targets" ] && return 1
            echo "$targets"
            ;;
        "random_")
            # random_ uses forget percentages as "targets"
            echo "${FORGET_PERC_VALUES[*]}"
            ;;
    esac
}

# Skip combinations that don't make sense (e.g. subclass on non-Cifar20)
is_applicable() {
    local strategy=$1 dataset=$2
    case "$strategy" in
        "fullclass") [[ "$dataset" =~ ^(Cifar20|Cifar100|PinsFaceRecognition|Caltech101)$ ]] ;;
        "subclass")  [[ "$dataset" == "Cifar20" ]] ;;
        "random_")   [[ "$dataset" =~ ^(Cifar10|PinsFaceRecognition|Caltech101)$ ]] ;;
    esac
}

# ==============================================================================
# Main grid loop: for each (seed × strategy × dataset × model × forget_target),
# invoke MAIN.sh sequentially.
# ==============================================================================
WORKER="$(dirname "$(realpath "$0")")/MAIN.sh"
if [ ! -x "$WORKER" ]; then
    echo "ERROR: worker script not found or not executable: $WORKER"
    exit 1
fi

CELL_COUNT=0
TOTAL_CELLS=0

# First pass: count cells (for progress reporting)
for SEED in "${SEED_VALUES[@]}"; do
    for STRATEGY in "${STRATEGY_VALUES[@]}"; do
        for DATASET in "${DATASET_VALUES[@]}"; do
            is_applicable "$STRATEGY" "$DATASET" || continue
            TARGETS=$(enumerate_targets "$STRATEGY" "$DATASET") || continue
            for MODEL in "${MODEL_VALUES[@]}"; do
                for _ in $TARGETS; do
                    TOTAL_CELLS=$((TOTAL_CELLS + 1))
                done
            done
        done
    done
done

echo "Total cells to run: $TOTAL_CELLS"
echo ""

# Second pass: execute
for SEED in "${SEED_VALUES[@]}"; do
    for STRATEGY in "${STRATEGY_VALUES[@]}"; do
        for DATASET in "${DATASET_VALUES[@]}"; do
            is_applicable "$STRATEGY" "$DATASET" || continue
            TARGETS=$(enumerate_targets "$STRATEGY" "$DATASET") || continue
            for MODEL in "${MODEL_VALUES[@]}"; do
                for TARGET in $TARGETS; do
                    CELL_COUNT=$((CELL_COUNT + 1))
                    echo ""
                    echo "##############################################################"
                    echo "Cell [$CELL_COUNT/$TOTAL_CELLS]: seed=$SEED strategy=$STRATEGY"
                    echo "                              dataset=$DATASET model=$MODEL target=$TARGET"
                    echo "##############################################################"

                    bash "$WORKER" "$SEED" "$STRATEGY" "$DATASET" "$MODEL" "$TARGET"
                done
            done
        done
    done
done

echo ""
echo "=============================================="
echo "ALL CELLS COMPLETED ($CELL_COUNT/$TOTAL_CELLS)"
echo "=============================================="
