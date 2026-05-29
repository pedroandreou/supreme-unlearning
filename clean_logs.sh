#!/bin/bash

# Function to get available precisions in a directory
get_available_precisions() {
	local base_path=$1
	if [ -d "$base_path" ]; then
		ls "$base_path" | grep "^precision_" | sed 's/precision_//'
	fi
}

# Function to list and delete specific seed directories.
# Walks the full hierarchy under precision_<p>/ since seed directories now live
# several levels deep:
#   <base>/precision_<p>/<N>gpus/[no_dist|dist_<strat>]/train_seed_<X>/...
# Legacy bare "seed_<X>" directories (from older runs) are also matched.
delete_seed_directories() {
	local base_path=$1
	local precision=$2
	local precision_root="${base_path}/precision_${precision}"

	if [ ! -d "$precision_root" ]; then
		echo -e "\n${precision_root} directory not found."
		return
	fi

	# Collect unique seed numbers found anywhere under precision_<p>/.
	# Match both current ("train_seed_<X>") and legacy ("seed_<X>") layouts.
	readarray -t available_seeds < <(
		find "$precision_root" -type d \( -name "train_seed_*" -o -name "seed_*" \) 2>/dev/null \
			| sed -E 's|.*/(train_)?seed_||' \
			| sort -u
	)

	if [ ${#available_seeds[@]} -eq 0 ]; then
		echo -e "\nNo seed directories found under ${precision_root}"
		return
	fi

	echo -e "\nAvailable seeds under ${precision_root}:"
	printf '%s\n' "${available_seeds[@]}"

	while true; do
		echo
		read -p "Enter seed number to delete (or 'all' for all seeds, 'n' to skip): " seed_answer

		case ${seed_answer} in
		all | ALL)
			echo -e "\nDeleting all seeds in precision_${precision}..."
			rm -rf "$precision_root"
			echo "All seeds deleted for precision_${precision}."
			break
			;;
		n | N)
			echo -e "\nSkipping deletion for precision_${precision}."
			break
			;;
		*)
			if [[ " ${available_seeds[*]} " =~ " ${seed_answer} " ]]; then
				echo -e "\nDeleting all train_seed_${seed_answer} / seed_${seed_answer} directories under precision_${precision}..."
				find "$precision_root" -type d \( -name "train_seed_${seed_answer}" -o -name "seed_${seed_answer}" \) -prune -exec rm -rf {} +
				echo "Seed ${seed_answer} deleted for precision_${precision}."
				break
			else
				echo -e "\nInvalid seed number. Please choose from the available seeds:"
				printf '%s\n' "${available_seeds[@]}"
			fi
			;;
		esac
	done
}

echo -e "\n=== Output Log Files ==="
# Ask about deleting output_log_files contents (except .gitkeep)
if [ -d "logs/output_log_files" ]; then
	read -p "Do you want to delete the contents of the output_log_files directory (except .gitkeep)? (y/n): " answer
	case ${answer:0:1} in
	y | Y)
		echo -e "\nDeleting contents of output_log_files directory (preserving .gitkeep)..."
		find logs/output_log_files -type f ! -name '.gitkeep' -delete
		echo "Output log files contents deleted (kept .gitkeep)."
		;;
	*)
		echo -e "\nKeeping output_log_files contents."
		;;
	esac
else
	echo -e "\nlogs/output_log_files directory not found."
fi

echo -e "\n=== Dataset Distributions ==="
# Ask about deleting dataset_distributions directory
if [ -d "logs/dataset_distributions" ]; then
	read -p "Do you want to delete the dataset_distributions directory and all its contents? (y/n): " answer
	case ${answer:0:1} in
	y | Y)
		echo -e "\nDeleting dataset_distributions directory..."
		rm -rf logs/dataset_distributions
		echo "Dataset distributions directory deleted."
		;;
	*)
		echo -e "\nKeeping dataset_distributions directory."
		;;
	esac
else
	echo -e "\nlogs/dataset_distributions directory not found."
fi

echo -e "\n=== Training Models ==="
# Get available precisions in training directory
available_training_precisions=($(get_available_precisions "logs/training"))

if [ ${#available_training_precisions[@]} -eq 0 ]; then
	echo -e "\nNo precision directories found in logs/training"
else
	# Handle each available precision
	for precision in "${available_training_precisions[@]}"; do
		echo -e "\nFound precision_${precision} in training directory"

		# First ask about processed_datasets if it exists
		if [ -d "logs/training/precision_${precision}/processed_datasets" ]; then
			read -p "Do you want to delete the processed_datasets for precision_${precision}? (y/n): " answer
			case ${answer:0:1} in
			y | Y)
				echo -e "\nDeleting processed_datasets for precision_${precision}..."
				rm -rf "logs/training/precision_${precision}/processed_datasets"
				echo "Processed datasets deleted for precision_${precision}."
				;;
			*)
				echo -e "\nKeeping processed_datasets for precision_${precision}."
				;;
			esac
		fi

		# Then ask about the entire precision directory
		read -p "Do you want to delete all training models for precision_${precision}? (y/n): " answer
		case ${answer:0:1} in
		y | Y)
			echo -e "\nDeleting training models for precision_${precision}..."
			rm -rf "logs/training/precision_${precision}"
			echo "Training models deleted for precision_${precision}."
			;;
		*)
			echo -e "\nKeeping training models for precision_${precision}."
			;;
		esac
	done
fi

echo -e "\n=== Unlearning Models ==="
# Get available precisions in unlearning directory
available_unlearning_precisions=($(get_available_precisions "logs/unlearning"))

if [ ${#available_unlearning_precisions[@]} -eq 0 ]; then
	echo -e "\nNo precision directories found in logs/unlearning"
else
	# Handle each available precision
	for precision in "${available_unlearning_precisions[@]}"; do
		echo -e "\nFound precision_${precision} in unlearning directory"
		delete_seed_directories "logs/unlearning" "${precision}"
	done
fi

echo -e "\n=== Wandb Logs ==="
# Wandb writes its local cache to <repo-root>/wandb/ (W&B's default = CWD,
# which is the project root for both run_local.sh and SLURM submissions).
if [ -d "wandb" ]; then
	read -p "Do you want to delete the wandb directory and all its contents? (y/n): " answer
	case ${answer:0:1} in
	y | Y)
		echo -e "\nDeleting wandb directory..."
		rm -rf wandb
		echo "Wandb directory deleted."
		;;
	*)
		echo -e "\nKeeping wandb directory."
		;;
	esac
else
	echo -e "\nWandb directory not found."
fi

echo -e "\nCleanup completed.\n"
