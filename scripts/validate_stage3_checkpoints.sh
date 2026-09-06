#!/usr/bin/env bash
# Evaluate existing checkpoints only. Requires an isolated checkout and a CUDA
# environment with the repository's dependencies, including DeepSpeed.
set -euo pipefail
: "${STAGE3_ARTIFACTS:?Path to artifact tree containing the scenario directory}"
: "${STAGE3_LOG_SUFFIX:?Relative LOG_DIR within that artifact tree}"
: "${STAGE3_WEIGHT:?Original model checkpoint}"
: "${CUDA_VISIBLE_DEVICES:?Select idle GPUs explicitly}"
STAGE3_PYTHON=${STAGE3_PYTHON:-python}
STAGE3_CASES=${STAGE3_CASES:-single ddp fsdp zero1 zero2 zero3}
STAGE3_PRECISION=${STAGE3_PRECISION:-bf16-true}
STAGE3_SCENARIO=${STAGE3_SCENARIO:-random_}
case "$STAGE3_SCENARIO" in
random_) stage3_scenario_args=(-classes "${STAGE3_CLASSES:-10}" -forget_perc "${STAGE3_FORGET_PERC:-0.01}") ;;
fullclass) stage3_scenario_args=(-classes "${STAGE3_CLASSES:?Set class count}" -forget_class_name "${STAGE3_FORGET_CLASS:?Set forget class}") ;;
subclass) stage3_scenario_args=(-superclasses "${STAGE3_SUPERCLASSES:?Set superclass count}" -subclasses "${STAGE3_SUBCLASSES:?Set subclass count}" -forget_subclass_name "${STAGE3_FORGET_CLASS:?Set forget subclass}") ;;
*)
	echo "Unknown scenario: $STAGE3_SCENARIO"
	exit 2
	;;
esac
STAGE3_OUTPUT=$(mktemp -d /tmp/supreme-stage3-validation-XXXXXX)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PERFORM_EVALUATION=true WANDB_LOG_EVALUATION=false LOG_PER_PROCESS_DATA=true
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
"$STAGE3_PYTHON" -c 'import supreme, torch, lightning; print(supreme.__file__, torch.__version__, lightning.__version__)'
echo "Validation output: $STAGE3_OUTPUT"
IFS=',' read -ra stage3_devices <<<"$CUDA_VISIBLE_DEVICES"
for stage3_case in $STAGE3_CASES; do
	stage3_path="$STAGE3_OUTPUT/$stage3_case"
	mkdir -p "$stage3_path"
	rsync -a --exclude='*_eval_results.json' "$STAGE3_ARTIFACTS/" "$stage3_path/"
	export LOG_DIR="$stage3_path/$STAGE3_LOG_SUFFIX"
	stage3_nproc=${#stage3_devices[@]}
	stage3_strategy=ddp
	stage3_extra=()
	case "$stage3_case" in
	single) stage3_nproc=1 ;;
	ddp | fsdp) stage3_strategy=$stage3_case ;;
	zero[123])
		stage3_strategy=deepspeed
		stage3_extra=(-deepspeed_stage "${stage3_case#zero}")
		;;
	*)
		echo "Unknown case: $stage3_case"
		exit 2
		;;
	esac
	stage3_visible=$CUDA_VISIBLE_DEVICES
	if [ "$stage3_nproc" = 1 ]; then stage3_visible=${stage3_devices[0]}; fi
	echo "Running $stage3_case ($stage3_nproc ranks, $STAGE3_PRECISION)"
	set +e
	CUDA_VISIBLE_DEVICES="$stage3_visible" timeout --signal=TERM --kill-after=15s 600s \
		"$STAGE3_PYTHON" -m supreme.utils.fabric.launch_single_node \
		--nproc-per-node="$stage3_nproc" \
		src/supreme/utils/unlearning/unlearn_main.py \
		-method "${STAGE3_METHOD:-finetune}" \
		-eval_metrics "${STAGE3_METRICS:-accuracy,zrf,jsdiv,membership_inference_attack,activation_distance,layerwise_distance,completeness,time}" \
		-net "${STAGE3_MODEL:-ResNet18}" -dataset "${STAGE3_DATASET:-Cifar10}" \
		-type_of_unlearning_strategy "$STAGE3_SCENARIO" "${stage3_scenario_args[@]}" \
		-seed "${STAGE3_SEED:-260}" -precision "$STAGE3_PRECISION" \
		-weight_path "$STAGE3_WEIGHT" -distributed_strategy "$stage3_strategy" \
		"${stage3_extra[@]}" "$@" >"$stage3_path/run.log" 2>&1
	stage3_status=$?
	set -e
	echo "$stage3_case exit=$stage3_status log=$stage3_path/run.log"
	if [ "$stage3_status" != 0 ]; then
		tail -70 "$stage3_path/run.log"
		exit "$stage3_status"
	fi
done
echo "All checkpoint evaluation cases passed: $STAGE3_OUTPUT"
if [[ " $STAGE3_CASES " == *" single "* && $STAGE3_CASES != "single" ]]; then
	"$STAGE3_PYTHON" scripts/compare_stage3_results.py "$STAGE3_OUTPUT" \
		--atol "${STAGE3_COMPARE_ATOL:-0.00001}"
fi
