#!/usr/bin/env bash
# Bounded numerical/layout/failure matrix. Use explicitly selected shared GPUs
# only when there is sufficient memory headroom; this is not a speed benchmark.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?Select devices explicitly}"
STAGE3_PYTHON=${STAGE3_PYTHON:-python}
STAGE3_PROBE_ENTRY=${STAGE3_PROBE_ENTRY:-scripts/probe_stage3_distributed.py}
STAGE3_PROBE_WORLDS=${STAGE3_PROBE_WORLDS:-2 4}
STAGE3_PROBE_PRECISIONS=${STAGE3_PROBE_PRECISIONS:-32-true bf16-true bf16-mixed 16-true 16-mixed}
STAGE3_PROBE_MODES=${STAGE3_PROBE_MODES:-per_device}
STAGE3_PROBE_FAILURES=${STAGE3_PROBE_FAILURES:-true}
STAGE3_CASES=${STAGE3_CASES:-ddp fsdp zero1 zero2 zero3}
stage3_output=$(mktemp -d /tmp/supreme-stage3-pinned-probes-XXXXXX)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 DATALOADER_NUM_WORKERS=0
echo "Probe output: $stage3_output"
"$STAGE3_PYTHON" -c 'import json,sys,torch,lightning,deepspeed,supreme; print(json.dumps(dict(python=sys.version,torch=torch.__version__,fabric=lightning.__version__,deepspeed=deepspeed.__version__,repository=supreme.__file__)))' >"$stage3_output/versions.log" 2>&1
tail -1 "$stage3_output/versions.log"
IFS=',' read -ra stage3_devices <<<"$CUDA_VISIBLE_DEVICES"
for stage3_world in $STAGE3_PROBE_WORLDS; do
	if ! [[ $stage3_world =~ ^[1-9][0-9]*$ ]] || ((stage3_world < 2 || stage3_world > ${#stage3_devices[@]})); then
		echo "Invalid probe world size: $stage3_world"
		exit 2
	fi
	stage3_visible=$(
		IFS=,
		echo "${stage3_devices[*]:0:stage3_world}"
	)
	for stage3_case in $STAGE3_CASES; do
		stage3_args=(--strategy "$stage3_case")
		case "$stage3_case" in zero[123]) stage3_args=(--strategy deepspeed --stage "${stage3_case#zero}") ;; esac
		for stage3_precision in $STAGE3_PROBE_PRECISIONS; do
			for stage3_mode in $STAGE3_PROBE_MODES; do
				stage3_batch=4
				[ "$stage3_mode" != global ] || stage3_batch=$((stage3_world * 2))
				stage3_name="${stage3_world}gpu-${stage3_case}-${stage3_precision}-${stage3_mode}"
				set +e
				CUDA_VISIBLE_DEVICES="$stage3_visible" timeout --kill-after=15s 180s "$STAGE3_PYTHON" \
					-m supreme.utils.fabric.launch_single_node --nproc-per-node="$stage3_world" \
					"$STAGE3_PROBE_ENTRY" "${stage3_args[@]}" --precision "$stage3_precision" \
					--samples 37 --batch-size "$stage3_batch" --batch-size-mode "$stage3_mode" \
					>"$stage3_output/$stage3_name.log" 2>&1
				stage3_status=$?
				set -e
				echo "$stage3_name exit=$stage3_status"
				if [ "$stage3_status" != 0 ]; then
					tail -70 "$stage3_output/$stage3_name.log"
					exit "$stage3_status"
				fi
				stage3_passes=$(grep -o 'STAGE3_PROBE_PASS rank=[0-9]*' "$stage3_output/$stage3_name.log" | sort -u | wc -l)
				if [ "$stage3_passes" -ne "$stage3_world" ]; then
					echo "Missing rank completion markers"
					exit 1
				fi
			done
		done
		if [ "$STAGE3_PROBE_FAILURES" = true ]; then
			for stage3_failure in rank mia; do
				stage3_name="${stage3_world}gpu-${stage3_case}-${stage3_failure}-failure"
				stage3_failure_args=(--fail-rank "$((stage3_world - 1))")
				stage3_failure_message='Injected Stage 3 rank-local failure'
				if [ "$stage3_failure" = mia ]; then
					stage3_failure_args=(--fail-mia-fit)
					stage3_failure_message='Injected Stage 3 MIA classifier failure'
				fi
				set +e
				CUDA_VISIBLE_DEVICES="$stage3_visible" timeout --kill-after=15s 90s "$STAGE3_PYTHON" \
					-m supreme.utils.fabric.launch_single_node --nproc-per-node="$stage3_world" \
					"$STAGE3_PROBE_ENTRY" "${stage3_args[@]}" --precision bf16-true \
					"${stage3_failure_args[@]}" >"$stage3_output/$stage3_name.log" 2>&1
				stage3_status=$?
				set -e
				echo "$stage3_name exit=$stage3_status (expected worker failure)"
				if [ "$stage3_status" = 0 ] || [ "$stage3_status" = 124 ] || [ "$stage3_status" = 137 ]; then
					echo "Failure test either succeeded unexpectedly or timed out"
					exit 1
				fi
				grep -q "$stage3_failure_message" "$stage3_output/$stage3_name.log"
			done
		fi
	done
done
echo "All requested probes passed: $stage3_output"
