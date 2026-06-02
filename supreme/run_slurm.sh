#!/bin/bash
# ==============================================================================
# SUPREME Pipeline - SLURM Dispatcher
# ==============================================================================
# Builds the experiment grid (seed × strategy × dataset × model × forget_target)
# and submits one SLURM array task per cell, each invoking supreme/MAIN.sh.
#
# Run from the login node:
#   ./run_slurm.sh --dry-run
#   ./run_slurm.sh --training-seeds 260,261,262
#   ./run_slurm.sh --strategies random_ --datasets Cifar10 --models ResNet18
#
# Options:
#   --training-seeds LIST  Comma-separated training seeds (default: 260-269)
#   --strategies LIST      Comma-separated strategies (default: fullclass,subclass,random_)
#   --datasets LIST        Comma-separated datasets (default: all)
#   --models LIST          Comma-separated models (default: ResNet18,ViT)
#   --methods LIST         Comma-separated unlearning methods (default: full set)
#   --metrics LIST         Comma-separated eval metrics (default: full set)
#   --forget-percs LIST    Comma-separated forget percentages for random_ (default: 0.001,...,0.10)
#   --fullclass-classes "c1 c2 ..."   Override fullclass forget targets
#   --subclass-classes  "c1 c2 ..."   Override subclass forget targets
#   --gpus N               GPUs per cell (default: 1)
#   --time HH:MM:SS        Time limit per cell (default: 48:00:00)
#   --partition NAME       SLURM partition (default: gpu)
#   --account NAME         SLURM account (default: $SLURM_ACCOUNT, from .env)
#   --max-concurrent N     Max concurrent array tasks (default: 12)
#   --wandb-prefix STR     WandB project prefix (default: R32)
#   --precision MODE       Precision (default: 32-true)
#   --force-retraining     Force retraining even if checkpoint exists
#   --force-rerun          Force re-execution ignoring existing W&B results and unlearning checkpoints
#   --cleanup-checkpoints  Delete method checkpoints after evaluation
#   --unlearning-seeds J   Space-separated unlearning-seed indices (default: "0", i.e. J=1)
#   --evaluation-seeds K   Space-separated evaluation-seed indices (default: "0", i.e. K=1)
#   --dry-run              Show jobs without submitting
# ==============================================================================

set -e

# ==============================================================================
# Defaults
# ==============================================================================
DEFAULT_SEEDS="260,261,262,263,264,265,266,267,268,269"
DEFAULT_STRATEGIES="fullclass,subclass,random_"
DEFAULT_DATASETS="Cifar10,Cifar20,Cifar100,PinsFaceRecognition,Caltech101"
DEFAULT_MODELS="ResNet18,ViT"
DEFAULT_METHODS="retrain,original,finetune,bad_teacher,random_labeling,unsir,ssd,lfssd,assd,scrub,jit"
DEFAULT_EVAL_METRICS="accuracy,activation_distance,completeness,jsdiv,layerwise_distance,time,membership_inference_attack"
DEFAULT_FORGET_PERCS="0.001,0.005,0.01,0.05,0.10"

FULLCLASS_CIFAR20="vehicle2 veg people electrical_devices natural_scenes"
FULLCLASS_CIFAR100="rocket mushroom baby lamp sea"
FULLCLASS_PINSFACE="1 10 20 30 40"
FULLCLASS_CALTECH101="airplanes car_side chair elephant lamp"
SUBCLASS_CIFAR20="rocket mushroom baby lamp sea"

SEEDS="$DEFAULT_SEEDS"
STRATEGIES="$DEFAULT_STRATEGIES"
DATASETS="$DEFAULT_DATASETS"
MODELS="$DEFAULT_MODELS"
METHODS="$DEFAULT_METHODS"
EVAL_METRICS="$DEFAULT_EVAL_METRICS"
FORGET_PERCS="$DEFAULT_FORGET_PERCS"
FULLCLASS_CLASSES_OVERRIDE=""
SUBCLASS_CLASSES_OVERRIDE=""
GPUS=1
TIME_LIMIT="48:00:00"
PARTITION="gpu"
# SLURM account: from the SLURM_ACCOUNT env var, or the SLURM_ACCOUNT entry in the
# repo-root .env if present. Override per-run with --account.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SLURM_ACCOUNT:-}" && -f "$_REPO_ROOT/.env" ]]; then
    SLURM_ACCOUNT="$(grep -E '^SLURM_ACCOUNT=' "$_REPO_ROOT/.env" | head -1 | cut -d= -f2- | tr -d '"')"
fi
ACCOUNT="${SLURM_ACCOUNT:-}"
MAX_CONCURRENT=12
WANDB_PROJECT_PREFIX="R32"
PRECISION="32-true"
FORCE_RETRAINING=false
FORCE_RERUN=false
CLEANUP_CHECKPOINTS=false
UNLEARNING_SEEDS_J="0"
EVALUATION_SEEDS_K="0"
DRY_RUN=false

# ==============================================================================
# Parse CLI flags
# ==============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --training-seeds)      SEEDS="$2"; shift 2 ;;
        --strategies)          STRATEGIES="$2"; shift 2 ;;
        --datasets)            DATASETS="$2"; shift 2 ;;
        --models)              MODELS="$2"; shift 2 ;;
        --methods)             METHODS="$2"; shift 2 ;;
        --metrics)             EVAL_METRICS="$2"; shift 2 ;;
        --forget-percs)        FORGET_PERCS="$2"; shift 2 ;;
        --fullclass-classes)   FULLCLASS_CLASSES_OVERRIDE="$2"; shift 2 ;;
        --subclass-classes)    SUBCLASS_CLASSES_OVERRIDE="$2"; shift 2 ;;
        --gpus)                GPUS="$2"; shift 2 ;;
        --time)                TIME_LIMIT="$2"; shift 2 ;;
        --partition)           PARTITION="$2"; shift 2 ;;
        --account)             ACCOUNT="$2"; shift 2 ;;
        --max-concurrent)      MAX_CONCURRENT="$2"; shift 2 ;;
        --wandb-prefix)        WANDB_PROJECT_PREFIX="$2"; shift 2 ;;
        --precision)           PRECISION="$2"; shift 2 ;;
        --force-retraining)    FORCE_RETRAINING=true; shift ;;
        --force-rerun)         FORCE_RERUN=true; shift ;;
        --cleanup-checkpoints) CLEANUP_CHECKPOINTS=true; shift ;;
        --unlearning-seeds)    UNLEARNING_SEEDS_J="$2"; shift 2 ;;
        --evaluation-seeds)    EVALUATION_SEEDS_K="$2"; shift 2 ;;
        --dry-run)             DRY_RUN=true; shift ;;
        --help|-h)
            grep -E "^# " "$0" | head -50
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$SLURM_JOB_ID" ]; then
    echo "ERROR: This script must be run from the login node, not inside a SLURM job."
    exit 1
fi

if [[ -z "$ACCOUNT" && "$DRY_RUN" != true ]]; then
    echo "ERROR: No SLURM account set. Add SLURM_ACCOUNT to .env or pass --account NAME."
    exit 1
fi

# ==============================================================================
# Build job configurations
# ==============================================================================
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=',' read -ra STRATEGY_ARRAY <<< "$STRATEGIES"
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
IFS=',' read -ra FORGET_PERC_ARRAY <<< "$FORGET_PERCS"

# Per-(dataset, strategy) forget targets
get_targets() {
    local strategy=$1 dataset=$2
    case "$strategy" in
        "fullclass")
            if [ -n "$FULLCLASS_CLASSES_OVERRIDE" ]; then
                echo "$FULLCLASS_CLASSES_OVERRIDE"
                return
            fi
            case "$dataset" in
                "Cifar20") echo "$FULLCLASS_CIFAR20" ;;
                "Cifar100") echo "$FULLCLASS_CIFAR100" ;;
                "PinsFaceRecognition") echo "$FULLCLASS_PINSFACE" ;;
                "Caltech101") echo "$FULLCLASS_CALTECH101" ;;
            esac
            ;;
        "subclass")
            if [ -n "$SUBCLASS_CLASSES_OVERRIDE" ]; then
                echo "$SUBCLASS_CLASSES_OVERRIDE"
                return
            fi
            case "$dataset" in
                "Cifar20") echo "$SUBCLASS_CIFAR20" ;;
            esac
            ;;
        "random_")
            echo "${FORGET_PERC_ARRAY[*]}"
            ;;
    esac
}

is_applicable() {
    local strategy=$1 dataset=$2
    case "$strategy" in
        "fullclass") [[ "$dataset" =~ ^(Cifar20|Cifar100|PinsFaceRecognition|Caltech101)$ ]] ;;
        "subclass")  [[ "$dataset" == "Cifar20" ]] ;;
        "random_")   [[ "$dataset" =~ ^(Cifar10|PinsFaceRecognition|Caltech101)$ ]] ;;
    esac
}

CONFIGS=()
for SEED in "${SEED_ARRAY[@]}"; do
    for STRATEGY in "${STRATEGY_ARRAY[@]}"; do
        for DATASET in "${DATASET_ARRAY[@]}"; do
            is_applicable "$STRATEGY" "$DATASET" || continue
            TARGETS=$(get_targets "$STRATEGY" "$DATASET")
            [ -z "$TARGETS" ] && continue
            for MODEL in "${MODEL_ARRAY[@]}"; do
                for TARGET in $TARGETS; do
                    JOB_NAME="supreme_${MODEL,,}_tseed${SEED}_${STRATEGY}_${DATASET,,}_${TARGET}"
                    # Format: JOB_NAME|SEED|STRATEGY|DATASET|MODEL|TARGET
                    CONFIGS+=("${JOB_NAME}|${SEED}|${STRATEGY}|${DATASET}|${MODEL}|${TARGET}")
                done
            done
        done
    done
done

TOTAL_JOBS=${#CONFIGS[@]}

if [ "$TOTAL_JOBS" -eq 0 ]; then
    echo "ERROR: No applicable (strategy, dataset, model, target) combinations to submit."
    exit 1
fi

# ==============================================================================
# Display configuration
# ==============================================================================
echo "=============================================="
echo "SUPREME - SLURM Submission"
echo "=============================================="
echo "Seeds (I=${#SEED_ARRAY[@]}): ${SEED_ARRAY[*]}"
echo "Strategies: ${STRATEGY_ARRAY[*]}"
echo "Datasets: ${DATASET_ARRAY[*]}"
echo "Models: ${MODEL_ARRAY[*]}"
echo "Methods: $METHODS"
echo "Eval Metrics: $EVAL_METRICS"
echo "Forget Percentages (random_): ${FORGET_PERC_ARRAY[*]}"
echo "Unlearning seed indices (J=$(echo $UNLEARNING_SEEDS_J | wc -w | tr -d ' ')): $UNLEARNING_SEEDS_J"
echo "Evaluation seed indices  (K=$(echo $EVALUATION_SEEDS_K | wc -w | tr -d ' ')): $EVALUATION_SEEDS_K"
echo "Total cells: $TOTAL_JOBS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "GPUs per cell: $GPUS"
echo "Time limit: $TIME_LIMIT"
echo "Partition: $PARTITION"
echo "Account: $ACCOUNT"
echo "WandB prefix: $WANDB_PROJECT_PREFIX"
echo "Precision: $PRECISION"
echo "Force retraining: $FORCE_RETRAINING"
echo "Force re-run: $FORCE_RERUN"
echo "Cleanup checkpoints: $CLEANUP_CHECKPOINTS"
echo "Dry run: $DRY_RUN"
echo "=============================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - first 5 and last 2 cells:"
    for i in "${!CONFIGS[@]}"; do
        if [ "$i" -lt 5 ] || [ "$i" -ge $((TOTAL_JOBS - 2)) ]; then
            echo "  [$((i+1))/$TOTAL_JOBS] ${CONFIGS[$i]}"
        elif [ "$i" -eq 5 ]; then
            echo "  ..."
        fi
    done
    echo ""
    echo "Would submit $TOTAL_JOBS jobs (max $MAX_CONCURRENT concurrent, $GPUS GPU(s) each)."
    exit 0
fi

# ==============================================================================
# Submit SLURM job array
# ==============================================================================
mkdir -p logs/output_log_files

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_FILE="logs/output_log_files/supreme_configs_${TIMESTAMP}.txt"
printf '%s\n' "${CONFIGS[@]}" > "$CONFIG_FILE"
echo "Config file: $CONFIG_FILE ($TOTAL_JOBS entries)"

ARRAY_SPEC="0-$((TOTAL_JOBS - 1))%${MAX_CONCURRENT}"

# Build sbatch --export string. Lists get embedded; commas are escaped to semicolons
# to avoid clashing with sbatch's own comma-separator, then converted back inside the job.
METHODS_E="${METHODS//,/;}"
METRICS_E="${EVAL_METRICS//,/;}"

EXPORT_VARS="ALL,CONFIG_FILE=${CONFIG_FILE}"
EXPORT_VARS+=",METHODS_E=${METHODS_E}"
EXPORT_VARS+=",METRICS_E=${METRICS_E}"
EXPORT_VARS+=",UNLEARNING_SEEDS_J=${UNLEARNING_SEEDS_J// /;}"
EXPORT_VARS+=",EVALUATION_SEEDS_K=${EVALUATION_SEEDS_K// /;}"
EXPORT_VARS+=",WANDB_PROJECT_PREFIX=${WANDB_PROJECT_PREFIX}"
EXPORT_VARS+=",PRECISION=${PRECISION}"
[ "$FORCE_RETRAINING"   = "true" ] && EXPORT_VARS+=",FORCE_RETRAINING=true"
[ "$FORCE_RERUN"        = "true" ] && EXPORT_VARS+=",FORCE_REUNLEARNING=true,FORCE_REEVALUATION=true"
[ "$CLEANUP_CHECKPOINTS" = "true" ] && EXPORT_VARS+=",CLEANUP_CHECKPOINTS_AFTER_EVAL=true"

JOB_ID=$(sbatch \
    --job-name="supreme" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks-per-node="$GPUS" \
    --gpus-per-node="$GPUS" \
    --time="$TIME_LIMIT" \
    --array="$ARRAY_SPEC" \
    --output="logs/output_log_files/supreme_%A_%a.out" \
    --error="logs/output_log_files/supreme_%A_%a.err" \
    --export="$EXPORT_VARS" \
    --wrap="
set -e

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

cd \$SLURM_SUBMIT_DIR

LINE_NUM=\$((SLURM_ARRAY_TASK_ID + 1))
CONFIG_LINE=\$(sed -n \"\${LINE_NUM}p\" \"\$CONFIG_FILE\")

if [ -z \"\$CONFIG_LINE\" ]; then
    echo \"ERROR: No config for array task \$SLURM_ARRAY_TASK_ID in \$CONFIG_FILE\"
    exit 1
fi

IFS='|' read -r JOB_NAME TRAINING_SEED STRATEGY DATASET MODEL FORGET_TARGET <<< \"\$CONFIG_LINE\"

echo \"=============================================\"
echo \"Array task \$SLURM_ARRAY_TASK_ID: \$JOB_NAME\"
echo \"=============================================\"

export TRAINING_SEED STRATEGY DATASET MODEL FORGET_TARGET
export METHODS=\"\${METHODS_E//;/,}\"
export EVAL_METRICS=\"\${METRICS_E//;/,}\"
export UNLEARNING_SEEDS_J=\"\${UNLEARNING_SEEDS_J//;/ }\"
export EVALUATION_SEEDS_K=\"\${EVALUATION_SEEDS_K//;/ }\"

bash supreme/MAIN.sh
" 2>&1 | awk '{print $4}')

echo ""
echo "=============================================="
echo "Submitted job array: $JOB_ID"
echo "Array spec: $ARRAY_SPEC"
echo "=============================================="
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f logs/output_log_files/supreme_${JOB_ID}_<task>.out"
echo ""
echo "Cancel with:"
echo "  scancel $JOB_ID"
echo "=============================================="
