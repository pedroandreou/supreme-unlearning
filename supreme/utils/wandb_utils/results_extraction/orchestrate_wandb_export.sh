#!/bin/bash
set -eo pipefail

################################################################################
### SCRIPT HEADER AND USAGE
################################################################################
# W&B Metrics Export Orchestration Script
#
# All configuration (models, datasets, strategies, seeds, prefixes) is loaded
# dynamically from: supreme/utils/wandb_utils/results_extraction/export_config.py
#
# Run with --help for full usage information with dynamically loaded values.
#
# Usage: ./orchestrate_wandb_export.sh [OPTIONS] [MODEL] [DATASET] [STRATEGY] [PRECISION] [SEED]
#
# OPTIONS:
#   --all-existing    Process all existing project combinations (fast, recommended)
#   --all-possible    Process all possible combinations (slower, may include non-existent projects)
#   --export          Export step only
#   --combine         Combine step only
#   --all-steps       Run all steps (default if no step flags provided)
#   --prefix PREFIX   Set experiment prefix (loaded from config by default)
#   --concurrent N    Run N export jobs concurrently (default: 1, recommended: 4-8)
#   --clean           Clean up directories and log files, then exit
#   --help            Show this help message with dynamic config values

################################################################################
### ENVIRONMENT SETUP
################################################################################

# Get this script's directory and project root
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." &>/dev/null && pwd)
export PYTHONPATH="$PROJECT_ROOT"

# Activate the project venv if one isn't already active. Honor $SUPREME_VENV,
# otherwise probe common names (the Makefile default is `unlearning`).
if [ -n "${VIRTUAL_ENV:-}" ]; then
	echo "=============================================="
	echo "Using already-active virtual environment: ${VIRTUAL_ENV}"
	echo "Python executable: $(which python)"
	echo "=============================================="
else
	VENV_PATH=""
	for _venv in "${SUPREME_VENV:-}" unlearning .venv gpu_env venv env; do
		if [ -n "$_venv" ] && [ -f "${PROJECT_ROOT}/${_venv}/bin/activate" ]; then
			VENV_PATH="${PROJECT_ROOT}/${_venv}"
			break
		fi
	done
	if [ -n "$VENV_PATH" ]; then
		echo "=============================================="
		echo "Activating virtual environment: ${VENV_PATH}"
		source "${VENV_PATH}/bin/activate"
		echo "Python executable: $(which python)"
		echo "Python version: $(python --version 2>&1)"
		echo "=============================================="
	else
		echo "=============================================="
		echo "WARNING: no virtual environment found under ${PROJECT_ROOT}"
		echo "Using system Python: $(which python)"
		echo "This may cause segmentation faults!"
		echo "=============================================="
	fi
fi

################################################################################
### CONFIGURATION FROM PYTHON CONFIG.PY
################################################################################

# Load all configuration from centralized config.py
# This ensures the shell script and Python code use the same values

# Get configured seeds
get_seeds() {
	python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import SEEDS
print(' '.join(map(str, SEEDS)))
"
}
CONFIGURED_SEEDS=($(get_seeds))

# Get default experiment prefix from config
get_default_prefix() {
	python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DEFAULT_EXPERIMENT_PREFIX
print(DEFAULT_EXPERIMENT_PREFIX)
"
}
DEFAULT_PREFIX=$(get_default_prefix)

# Get default precision from config
get_default_precision() {
	python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DEFAULT_PRECISION
print(DEFAULT_PRECISION)
"
}
DEFAULT_PRECISION=$(get_default_precision)

# Get available models from config
get_models() {
	python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import MODELS
print(' '.join(MODELS))
"
}

# Get available datasets from config
get_datasets() {
	python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DATASETS
print(' '.join(DATASETS))
"
}

# Get available prefixes from config
get_prefixes() {
	python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import EXPERIMENT_PREFIXES
print(' '.join(EXPERIMENT_PREFIXES))
"
}

################################################################################
### DIRECTORY AND FLAG INITIALIZATION
################################################################################

# Output directories (will be updated after parsing --prefix)
SUMMARY_DIR=""
COMBINED_DIR=""

# Control flags
EXPORT_ONLY=false
COMBINE_ONLY=false
ALL_STEPS=false
ALL_EXISTING=false
ALL_POSSIBLE=false
CLEAN_ONLY=false
ANY_FLAG_SET=false
CONCURRENT_JOBS=1  # Number of concurrent workers (1 = sequential)

# Common settings (use values from export_config.py)
PRECISION=${PRECISION:-"$DEFAULT_PRECISION"}
SEED=${SEED:-"0"}
EXPERIMENT_PREFIX=${EXPERIMENT_PREFIX:-"$DEFAULT_PREFIX"}

################################################################################
### HELPER FUNCTIONS
################################################################################

# Function to show help message (values loaded dynamically from export_config.py)
show_help() {
	# Get dynamic values from config
	local AVAILABLE_MODELS=$(get_models)
	local AVAILABLE_DATASETS=$(get_datasets)
	local AVAILABLE_PREFIXES=$(get_prefixes)

	cat <<EOF
Usage: ./orchestrate_wandb_export.sh [OPTIONS] [MODEL] [DATASET] [STRATEGY] [PRECISION] [SEED]

OPTIONS:
  --all-existing    Process all existing project combinations (fast, recommended)
  --all-possible    Process all possible combinations (slower, may include non-existent projects)
  --export          Export step only
  --combine         Combine step only
  --all-steps       Run export and combine steps (default if no step flags provided)
  --prefix PREFIX   Experiment prefix (default: $DEFAULT_PREFIX)
                    Available: $AVAILABLE_PREFIXES
  --concurrent N    Run N export jobs concurrently (default: 1 = sequential)
                    Recommended: 4-8 for faster exports (CPU/network bound, not GPU)
  --clean           Clean up directories and log files, then exit
  --help            Show this help message

EXAMPLES:
  ./orchestrate_wandb_export.sh --all-existing                           # Process all existing projects
  ./orchestrate_wandb_export.sh --all-existing --concurrent 4            # Process with 4 concurrent workers
  ./orchestrate_wandb_export.sh --all-existing --prefix R6_UNLEARNING   # Process R6 projects
  ./orchestrate_wandb_export.sh ResNet18 Cifar20 fullclass $DEFAULT_PRECISION 0    # Process specific combination
  ./orchestrate_wandb_export.sh --clean                                  # Clean up files
  ./orchestrate_wandb_export.sh --export ResNet18 Cifar20 fullclass     # Export only for specific combo

POSITIONAL ARGUMENTS (for specific combinations):
  MODEL      Model name
             Available: $AVAILABLE_MODELS
  DATASET    Dataset name
             Available: $AVAILABLE_DATASETS
  STRATEGY   Unlearning strategy (fullclass, subclass, random_)
  PRECISION  Precision setting (default: $DEFAULT_PRECISION)
  SEED       Random seed (default: 0)

CONFIGURATION:
  All values are loaded from: supreme/utils/wandb_utils/results_extraction/export_config.py
  Seeds configured: ${CONFIGURED_SEEDS[*]}
  Default prefix: $DEFAULT_PREFIX
  Default precision: $DEFAULT_PRECISION
EOF
}

# Function to run a specific model/dataset/strategy/class/seed combination
run_specific_combination() {
	local model=$1
	local dataset=$2
	local strategy=$3
	local class_or_perc=$4
	local precision=$5
	local seed=$6

	echo "========================================================================"
	echo "VALIDATING: $model, $dataset, $strategy, $class_or_perc, seed $seed"

	if [ "$strategy" == "random_" ]; then
		echo "Exporting metrics for $dataset, $model, $strategy, forget_perc: $class_or_perc (seed: $seed)"
		python -m supreme.utils.wandb_utils.results_extraction.wandb_metrics_exporter --export \
			-model "$model" \
			-dataset "$dataset" \
			-type_of_unlearning_strategy "$strategy" \
			-forget_perc "$class_or_perc" \
			-precision "$precision" \
			-seed "$seed" \
			--prefix "$EXPERIMENT_PREFIX" \
			--summary-dir "$SUMMARY_DIR" >>"${LOG_FILE}" 2>&1

		exit_code=$?
		if [ $exit_code -ne 0 ]; then
			echo "ERROR: Python script failed with exit code $exit_code for $model/$dataset/$strategy/$class_or_perc/seed$seed"
		fi
	else
		echo "Exporting metrics for $dataset, $model, $strategy, class: $class_or_perc (seed: $seed)"
		python -m supreme.utils.wandb_utils.results_extraction.wandb_metrics_exporter --export \
			-model "$model" \
			-dataset "$dataset" \
			-type_of_unlearning_strategy "$strategy" \
			-class_name "$class_or_perc" \
			-precision "$precision" \
			-seed "$seed" \
			--prefix "$EXPERIMENT_PREFIX" \
			--summary-dir "$SUMMARY_DIR" >>"${LOG_FILE}" 2>&1

		exit_code=$?
		if [ $exit_code -ne 0 ]; then
			echo "ERROR: Python script failed with exit code $exit_code for $model/$dataset/$strategy/$class_or_perc/seed$seed"
		fi
	fi
	echo "========================================================================"
}

# Function to run exports concurrently using xargs for reliable parallel execution
# Arguments: array of "model dataset strategy class_or_perc precision seed" strings
run_concurrent_exports() {
	local jobs_array=("$@")
	local total_jobs=${#jobs_array[@]}

	echo "Starting concurrent export with $CONCURRENT_JOBS workers for $total_jobs jobs..."

	# Create a temporary file to store job definitions
	local jobs_file=$(mktemp)
	
	# Write all jobs to the file (one per line)
	for job in "${jobs_array[@]}"; do
		echo "$job" >> "$jobs_file"
	done

	# Export variables needed by the worker function
	export EXPERIMENT_PREFIX
	export SUMMARY_DIR
	export LOG_DIR
	export PYTHONPATH

	# Define the worker function that processes a single job
	process_single_job() {
		local job="$1"
		read -r model dataset strategy class_or_perc precision seed <<<"$job"
		
		# Create per-job log file
		local job_log="${LOG_DIR}/export_${model}_${dataset}_${strategy}_${class_or_perc}_seed${seed}.log"
		
		echo "[$(date '+%H:%M:%S')] Starting: $model $dataset $strategy $class_or_perc seed$seed"
		
		if [ "$strategy" == "random_" ]; then
			python -m supreme.utils.wandb_utils.results_extraction.wandb_metrics_exporter --export \
				-model "$model" \
				-dataset "$dataset" \
				-type_of_unlearning_strategy "$strategy" \
				-forget_perc "$class_or_perc" \
				-precision "$precision" \
				-seed "$seed" \
				--prefix "$EXPERIMENT_PREFIX" \
				--summary-dir "$SUMMARY_DIR" >"$job_log" 2>&1
		else
			python -m supreme.utils.wandb_utils.results_extraction.wandb_metrics_exporter --export \
				-model "$model" \
				-dataset "$dataset" \
				-type_of_unlearning_strategy "$strategy" \
				-class_name "$class_or_perc" \
				-precision "$precision" \
				-seed "$seed" \
				--prefix "$EXPERIMENT_PREFIX" \
				--summary-dir "$SUMMARY_DIR" >"$job_log" 2>&1
		fi
		
		local exit_code=$?
		if [ $exit_code -eq 0 ]; then
			echo "[$(date '+%H:%M:%S')] Completed: $model $dataset $strategy $class_or_perc seed$seed"
		else
			echo "[$(date '+%H:%M:%S')] FAILED (exit $exit_code): $model $dataset $strategy $class_or_perc seed$seed"
		fi
	}
	export -f process_single_job

	# Check if GNU parallel is available (preferred)
	if command -v parallel &>/dev/null; then
		echo "Using GNU parallel for concurrent execution..."
		cat "$jobs_file" | parallel -j "$CONCURRENT_JOBS" --progress process_single_job {}
	else
		# Fallback to xargs (available on all systems)
		echo "Using xargs for concurrent execution (install GNU parallel for better progress reporting)..."
		cat "$jobs_file" | xargs -P "$CONCURRENT_JOBS" -I {} bash -c 'process_single_job "$@"' _ {}
	fi

	# Clean up
	rm -f "$jobs_file"

	# Aggregate all concurrent logs into the main log file
	if [ -d "$LOG_DIR" ]; then
		echo ""
		echo "Aggregating concurrent job logs into main log file..."
		echo "" >> "${LOG_FILE}"
		echo "========== CONCURRENT JOB LOGS ==========" >> "${LOG_FILE}"
		for log in "$LOG_DIR"/*.log; do
			if [ -f "$log" ]; then
				echo "--- $(basename "$log") ---" >> "${LOG_FILE}"
				cat "$log" >> "${LOG_FILE}"
				echo "" >> "${LOG_FILE}"
			fi
		done
		echo "========== END CONCURRENT JOB LOGS ==========" >> "${LOG_FILE}"
	fi

	echo ""
	echo "All $total_jobs export jobs completed"
}

################################################################################
### COMMAND-LINE ARGUMENT PARSING
################################################################################

while [[ $# -gt 0 ]]; do
	case $1 in
	--export)
		EXPORT_ONLY=true
		ANY_FLAG_SET=true
		shift
		;;
	--combine)
		COMBINE_ONLY=true
		ANY_FLAG_SET=true
		shift
		;;
	--all-steps)
		ALL_STEPS=true
		ANY_FLAG_SET=true
		shift
		;;
	--all-existing)
		ALL_EXISTING=true
		ANY_FLAG_SET=true
		shift
		;;
	--all-possible)
		ALL_POSSIBLE=true
		ANY_FLAG_SET=true
		shift
		;;
	--clean)
		CLEAN_ONLY=true
		ANY_FLAG_SET=true
		shift
		;;
	--prefix)
		EXPERIMENT_PREFIX="$2"
		shift 2
		;;
	--concurrent)
		CONCURRENT_JOBS="$2"
		if ! [[ "$CONCURRENT_JOBS" =~ ^[0-9]+$ ]] || [ "$CONCURRENT_JOBS" -lt 1 ]; then
			echo "Error: --concurrent requires a positive integer"
			exit 1
		fi
		shift 2
		;;
	--help)
		show_help
		exit 0
		;;
	*) # Positional arguments
		if [ -z "$MODEL_ARG" ]; then
			MODEL_ARG=$1
		elif [ -z "$DATASET_ARG" ]; then
			DATASET_ARG=$1
		elif [ -z "$STRATEGY_ARG" ]; then
			STRATEGY_ARG=$1
		elif [ -z "$PRECISION_ARG" ]; then
			PRECISION_ARG=$1
		elif [ -z "$SEED_ARG" ]; then
			SEED_ARG=$1
		else
			echo "Error: Unknown argument '$1'"
			show_help
			exit 1
		fi
		shift
		;;
	esac
done

################################################################################
### MAIN EXECUTION FLOW
################################################################################

### Set output directories based on prefix (R6 or R7)
# Extract short prefix name (e.g., "R7" from "R7_UNLEARNING")
PREFIX_SHORT="${EXPERIMENT_PREFIX%%_*}"
SUMMARY_DIR="$SCRIPT_DIR/wandb_metrics_summary_${PREFIX_SHORT}"
COMBINED_DIR="$SCRIPT_DIR/combined_results_${PREFIX_SHORT}"

echo "Using experiment prefix: $EXPERIMENT_PREFIX"
echo "Summary directory: $SUMMARY_DIR"
echo "Combined directory: $COMBINED_DIR"

### Clean up mode (if requested, clean and exit)
if [ "$CLEAN_ONLY" = true ]; then
	echo "==================================================================="
	echo "Cleaning up directories and log files"
	echo "==================================================================="

	if [ -d "$SUMMARY_DIR" ]; then
		echo "Removing $SUMMARY_DIR"
		rm -rf "$SUMMARY_DIR"
	fi

	if [ -d "$COMBINED_DIR" ]; then
		echo "Removing $COMBINED_DIR"
		rm -rf "$COMBINED_DIR"
	fi

	# Clean up any log files
	echo "Removing log files"
	rm -f ./export_*.log

	# Clean up concurrent log directories
	echo "Removing concurrent log directories"
	rm -rf ./logs_concurrent_*

	echo "Cleanup completed."
	exit 0
fi

### Determine operation mode
if [ "$ALL_EXISTING" = true ]; then
	MODE="all-existing"
elif [ "$ALL_POSSIBLE" = true ]; then
	MODE="all-possible"
elif [ -n "$MODEL_ARG" ] || [ -n "$DATASET_ARG" ] || [ -n "$STRATEGY_ARG" ]; then
	MODE="specific"
else
	echo "Error: Must specify either --all-existing, --all-possible, or provide MODEL/DATASET/STRATEGY parameters"
	show_help
	exit 1
fi

### Determine which steps to run
# If no step flags are provided, default to running all steps
if [ "$ANY_FLAG_SET" = false ] || ([ "$ALL_EXISTING" = true ] || [ "$ALL_POSSIBLE" = true ]) && [ "$EXPORT_ONLY" = false ] && [ "$COMBINE_ONLY" = false ]; then
	ALL_STEPS=true
fi

### Set final values for specific mode
if [ "$MODE" = "specific" ]; then
	MODEL=${MODEL_ARG:-"ResNet18"}
	DATASET=${DATASET_ARG:-"Cifar20"}
	STRATEGY=${STRATEGY_ARG:-"fullclass"}
	PRECISION=${PRECISION_ARG:-"32-true"}
	SEED=${SEED_ARG:-"0"}
fi

### Setup logging
# Create a temporary file to store the output logs
if [ "$MODE" = "specific" ]; then
	LOG_FILE=$(mktemp ./export_metrics_${DATASET}_${MODEL}.XXXXXX.log)
else
	LOG_FILE=$(mktemp ./export_${MODE//-/_}.XXXXXX.log)
fi
echo "Logging output to ${LOG_FILE}"

# Create log directory for concurrent execution
if [ "$CONCURRENT_JOBS" -gt 1 ]; then
	LOG_DIR="./logs_concurrent_${PREFIX_SHORT}_$(date +%Y%m%d_%H%M%S)"
	mkdir -p "$LOG_DIR"
	echo "Concurrent execution enabled with $CONCURRENT_JOBS workers"
	echo "Per-job logs will be saved to: $LOG_DIR"
fi

### Print execution plan
echo "==================================================================="
if [ "$MODE" = "specific" ]; then
	echo "Processing specific combination: $MODEL $DATASET $STRATEGY"
elif [ "$MODE" = "all-existing" ]; then
	echo "Processing all existing project combinations"
else
	echo "Processing all possible project combinations"
fi
echo "==================================================================="

################################################################################
### STEP 1: EXPORT METRICS FROM WANDB
################################################################################

if [ "$EXPORT_ONLY" = true ] || [ "$ALL_STEPS" = true ]; then
	echo "--- Running Export Step ---"

	### MODE 1: Process specific model/dataset/strategy combination
	if [ "$MODE" = "specific" ]; then
		# Special case: Random strategy uses forget percentages instead of classes
		if [ "$STRATEGY" == "random_" ]; then
			# Get forget percentages from config.py dynamically
			FORGET_PERCS_STR=$(python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DATASET_CLASSES
key = '${DATASET}_${STRATEGY}'
if key in DATASET_CLASSES:
    print(' '.join(DATASET_CLASSES[key]))
else:
    print(f'ERROR: No configuration found for {key}', file=__import__('sys').stderr)
    exit(1)
" 2>&1)

			# Check if the Python command succeeded
			if [ $? -ne 0 ]; then
				echo "$FORGET_PERCS_STR"
				exit 1
			fi

			read -r -a FORGET_PERCS <<<"$FORGET_PERCS_STR"
			echo "Forget percentages to process for $DATASET $STRATEGY: ${FORGET_PERCS[@]}"

			for SEED in "${CONFIGURED_SEEDS[@]}"; do
				for FORGET_PERC in "${FORGET_PERCS[@]}"; do
					run_specific_combination "$MODEL" "$DATASET" "$STRATEGY" "$FORGET_PERC" "$PRECISION" "$SEED"
				done
			done
		else
			# Get dictionary name dynamically from centralized config
			DICT_NAME=$(python -c "
import supreme.utils.project_config as project_config
try:
    dict_name = project_config.get_dict_name_for_dataset('$DATASET', '$STRATEGY')
    print(dict_name)
except ValueError as e:
    print(f'ERROR: {e}', file=__import__('sys').stderr)
    exit(1)
" 2>&1)

			# Check if the Python command succeeded
			if [ $? -ne 0 ]; then
				echo "$DICT_NAME"
				exit 1
			fi

			if [ -n "$DICT_NAME" ]; then
				CLASSES_STR=$(python -c "import supreme.utils.project_config as project_config; print(' '.join(getattr(project_config, '$DICT_NAME').keys()))" 2>/dev/null)
				if [ -z "$CLASSES_STR" ]; then
					echo "Error: Could not retrieve class list for dataset '$DATASET' from supreme/utils/project_config.py."
					exit 1
				fi
				read -r -a CLASSES <<<"$CLASSES_STR"

				echo "Classes to process for $DATASET $STRATEGY: ${CLASSES[@]}"

				for SEED in "${CONFIGURED_SEEDS[@]}"; do
					for CLASS in "${CLASSES[@]}"; do
						run_specific_combination "$MODEL" "$DATASET" "$STRATEGY" "$CLASS" "$PRECISION" "$SEED"
					done
				done
			fi
		fi

	### MODE 2: Process all existing projects (from config.py EXISTING_PROJECTS)
	elif [ "$MODE" = "all-existing" ]; then
		# Get existing project combinations dynamically from config.py
		# Each tuple is converted to a space-separated string
		mapfile -t EXISTING_PROJECTS < <(python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import EXISTING_PROJECTS
for project in EXISTING_PROJECTS:
    print(' '.join(project))
")

		echo "Processing ${#EXISTING_PROJECTS[@]} existing project combinations"

		# Build list of all jobs - ONE JOB PER PROJECT (not per seed!)
		# The Python exporter fetches ALL seeds for a project in one run,
		# so we should NOT multiply by seeds (that would cause race conditions
		# with multiple processes writing to the same Excel files)
		EXPORT_JOBS=()
		for PROJECT in "${EXISTING_PROJECTS[@]}"; do
			read -r MODEL DATASET STRATEGY CLASS PROJ_PRECISION <<<"$PROJECT"
			# Use project-specific precision if provided, otherwise fall back to the global default
			EFFECTIVE_PRECISION=${PROJ_PRECISION:-$PRECISION}
			# Pass first seed as placeholder (Python fetches all seeds anyway)
			EXPORT_JOBS+=("$MODEL $DATASET $STRATEGY $CLASS $EFFECTIVE_PRECISION ${CONFIGURED_SEEDS[0]}")
		done

		echo "Total export jobs: ${#EXPORT_JOBS[@]} projects (each fetches all ${#CONFIGURED_SEEDS[@]} seeds)"

		if [ "$CONCURRENT_JOBS" -gt 1 ]; then
			# Run exports concurrently
			run_concurrent_exports "${EXPORT_JOBS[@]}"
		else
			# Run exports sequentially (original behavior)
			for job in "${EXPORT_JOBS[@]}"; do
				read -r MODEL DATASET STRATEGY CLASS EFFECTIVE_PRECISION SEED <<<"$job"
				echo "Processing project: $MODEL $DATASET $STRATEGY $CLASS (all seeds)"
				run_specific_combination "$MODEL" "$DATASET" "$STRATEGY" "$CLASS" "$EFFECTIVE_PRECISION" "$SEED"
			done
		fi

	### MODE 3: Process all possible combinations (from config.py MODELS/DATASETS/etc.)
	elif [ "$MODE" = "all-possible" ]; then
		# Get all combinations dynamically from config.py

		# Get models array
		IFS=' ' read -r -a MODELS <<< "$(python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import MODELS
print(' '.join(MODELS))
")"

		# Get datasets array
		IFS=' ' read -r -a DATASETS <<< "$(python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DATASETS
print(' '.join(DATASETS))
")"

		# Get dataset classes as associative array
		declare -A DATASET_CLASSES
		while IFS='=' read -r key value; do
			DATASET_CLASSES["$key"]="$value"
		done < <(python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DATASET_CLASSES
for key, classes in DATASET_CLASSES.items():
    print(f'{key}={\" \".join(classes)}')
")

		# Get dataset strategies as associative array
		declare -A DATASET_STRATEGIES
		while IFS='=' read -r key value; do
			DATASET_STRATEGIES["$key"]="$value"
		done < <(python -c "
from supreme.utils.wandb_utils.results_extraction.export_config import DATASET_STRATEGIES
for dataset, strategies in DATASET_STRATEGIES.items():
    print(f'{dataset}={\" \".join(strategies)}')
")

		# Build list of all jobs - ONE JOB PER PROJECT (not per seed!)
		EXPORT_JOBS=()
		echo "Building job list for all possible combinations..."
		for MODEL in "${MODELS[@]}"; do
			for DATASET in "${DATASETS[@]}"; do
				STRATEGIES=(${DATASET_STRATEGIES[$DATASET]})
				for STRATEGY in "${STRATEGIES[@]}"; do
					CLASSES=(${DATASET_CLASSES["${DATASET}_${STRATEGY}"]})
					for CLASS in "${CLASSES[@]}"; do
						# Pass first seed as placeholder (Python fetches all seeds anyway)
						EXPORT_JOBS+=("$MODEL $DATASET $STRATEGY $CLASS $PRECISION ${CONFIGURED_SEEDS[0]}")
					done
				done
			done
		done

		echo "Total export jobs: ${#EXPORT_JOBS[@]} projects (each fetches all ${#CONFIGURED_SEEDS[@]} seeds)"

		if [ "$CONCURRENT_JOBS" -gt 1 ]; then
			# Run exports concurrently
			run_concurrent_exports "${EXPORT_JOBS[@]}"
		else
			# Run exports sequentially (original behavior)
			for job in "${EXPORT_JOBS[@]}"; do
				read -r MODEL DATASET STRATEGY CLASS JOB_PRECISION SEED <<<"$job"
				echo "Processing: $MODEL $DATASET $STRATEGY $CLASS (all seeds)"
				run_specific_combination "$MODEL" "$DATASET" "$STRATEGY" "$CLASS" "$JOB_PRECISION" "$SEED"
			done
		fi
	fi
fi

################################################################################
### STEP 2: COMBINE EXPORTED METRICS
################################################################################

if [ "$COMBINE_ONLY" = true ] || [ "$ALL_STEPS" = true ]; then
	echo "--- Running Combine Step ---"
	python -m supreme.utils.wandb_utils.results_extraction.wandb_metrics_exporter --combine \
		--prefix "$EXPERIMENT_PREFIX" \
		--summary-dir "$SUMMARY_DIR" \
		--combined-dir "$COMBINED_DIR" >>"${LOG_FILE}" 2>&1
fi

################################################################################
### COMPLETION MESSAGE
################################################################################

echo "==================================================================="
echo "Processing completed successfully!"
echo "Results available in:"
echo "  - Summaries: $SUMMARY_DIR"
echo "  - Combined: $COMBINED_DIR"
echo "==================================================================="
echo "Full logs available at: ${LOG_FILE}"
if [ "$CONCURRENT_JOBS" -gt 1 ] && [ -d "$LOG_DIR" ]; then
	echo "Individual job logs available in: ${LOG_DIR}"
fi