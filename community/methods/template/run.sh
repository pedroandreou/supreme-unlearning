#!/bin/bash
set -euo pipefail

############################################################################
# Hyperparameter tuning (optional)
############################################################################

# Document any search you ran to arrive at the final settings below.

############################################################################
# Final best parameters (required to reproduce your results)
############################################################################

# Replace METHOD with your registered method name and adjust the flags to the
# exact sweep that produces your numbers.
# See docs/script_arguments.md for the full flag reference.
METHOD="your_method"

bash src/supreme/run_local.sh \
	--gpu 0 \
	--models ViT \
	--methods "$METHOD" \
	--strategies random_ \
	--datasets Cifar10 \
	--forget-percs 0.01 \
	--training-seeds 260,261,262
