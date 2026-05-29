#!/bin/bash
# ==============================================================================
# Unified WandB Tools for Machine Unlearning Experiments
# ==============================================================================
# This script provides a unified interface for WandB management.
#
# MODES (ordered from informational to action commands):
#
#   INFORMATIONAL:
#     list-projects    List all expected project names
#     find-duplicates  Find duplicate runs across projects
#     generate-report  Generate detailed duplicate report (JSON)
#
#   ACTIONS:
#     sync             Sync offline WandB runs to the cloud
#     cleanup-empty    Delete runs with no evaluation metrics
#     delete-duplicates Delete identical duplicate runs
#
# USAGE:
#   ./wandb_tools.sh <mode> [options]
#
# SYNC MODE OPTIONS:
#   --clean               Sync conflicted runs as new runs
#   --delete-synced       Delete local directories after all syncs complete
#   --delete-immediately  Delete each directory immediately after successful sync
#
# PYTHON MANAGER OPTIONS (all other modes):
#   See: python wandb_manager.py <mode> --help
#
# EXAMPLES:
#   # List expected projects
#   ./wandb_tools.sh list-projects
#   ./wandb_tools.sh list-projects --strategy fullclass
#
#   # Find all duplicates
#   ./wandb_tools.sh find-duplicates --all
#
#   # Generate duplicate report
#   ./wandb_tools.sh generate-report -o report.json
#
#   # Sync offline runs
#   ./wandb_tools.sh sync
#   ./wandb_tools.sh sync --delete-immediately
#
#   # Find empty runs (dry run)
#   ./wandb_tools.sh cleanup-empty --project "R7_UNLEARNING_..."
#
#   # Delete empty runs
#   ./wandb_tools.sh cleanup-empty --project "..." --delete
#
#   # Delete identical duplicates
#   ./wandb_tools.sh delete-duplicates --all --confirm
# ==============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_MANAGER="${SCRIPT_DIR}/wandb_manager.py"

# Default wandb directory (relative to script location)
WANDB_DIR="${SCRIPT_DIR}/../../../wandb"

# ==============================================================================
# Help
# ==============================================================================
show_help() {
    head -53 "$0" | tail -50
    exit 0
}

# ==============================================================================
# Sync Mode Functions
# ==============================================================================
run_sync_mode() {
    local sync_args=()
    local delete_synced=false
    local delete_immediately=false

    # Parse sync-specific arguments
    for arg in "$@"; do
        case "$arg" in
            --clean)
                echo "Running with --clean option. Conflicted runs will be synced as new runs."
                sync_args+=(--clean --clean-force)
                ;;
            --delete-synced)
                echo "Local directories will be deleted after all syncs complete."
                delete_synced=true
                ;;
            --delete-immediately)
                echo "Local directories will be deleted immediately after each successful sync."
                delete_immediately=true
                ;;
            --help|-h)
                echo "Sync Mode Options:"
                echo "  --clean               Sync conflicted runs as new runs"
                echo "  --delete-synced       Delete all local directories after all syncs complete"
                echo "  --delete-immediately  Delete each local directory immediately after successful sync"
                exit 0
                ;;
        esac
    done

    # Login to wandb
    echo "Logging in to WandB..."
    wandb login

    # Check wandb directory
    if [ ! -d "$WANDB_DIR" ]; then
        echo "WandB directory not found: $WANDB_DIR"
        echo "Trying alternative path..."
        WANDB_DIR="${SCRIPT_DIR}/../../../wandb"
        if [ ! -d "$WANDB_DIR" ]; then
            echo "Error: Cannot find wandb directory"
            exit 1
        fi
    fi

    # Create temporary file to store directories
    local temp_file=$(mktemp)

    # Find run directories
    find "$WANDB_DIR" -name "run-*" -type d > "$temp_file"

    # Check if any runs found
    if [ ! -s "$temp_file" ]; then
        echo "No offline runs found in $WANDB_DIR"
        echo "Checking what's in the directory:"
        ls "$WANDB_DIR" 2>/dev/null | head -10 || echo "  (empty or not accessible)"
        rm "$temp_file"
        exit 0
    fi

    echo "Found $(wc -l < "$temp_file") offline run(s) to sync."
    echo ""

    # Arrays for tracking
    local synced_dirs=()
    local failed_count=0
    local success_count=0

    # Process each directory
    while read -r dir; do
        echo "================================"
        echo "Syncing directory: $dir"
        echo "================================"

        # Create temp file for output
        local sync_output_file=$(mktemp)

        # Run wandb sync
        if wandb sync "${sync_args[@]}" "$dir" > "$sync_output_file" 2>&1; then
            local sync_exit_code=0
        else
            local sync_exit_code=$?
        fi

        # Display output
        cat "$sync_output_file"

        # Check for 409 errors
        if grep -q "409" "$sync_output_file" || grep -q "previously created and deleted" "$sync_output_file"; then
            echo ""
            echo "SYNC FAILED: Run was previously deleted (409 conflict)"
            echo "   This run will NOT be deleted locally."
            if [ "${#sync_args[@]}" -eq 0 ]; then
                echo "   Tip: Use --clean flag to force sync as a new run"
            fi
            failed_count=$((failed_count + 1))
            rm "$sync_output_file"
            continue
        fi

        # Check for success
        if [ $sync_exit_code -eq 0 ] && (grep -q "done\." "$sync_output_file" || grep -q "Success!" "$sync_output_file" || grep -q "Syncing:" "$sync_output_file"); then
            echo ""
            echo "Successfully synced $dir"
            success_count=$((success_count + 1))

            if [ "$delete_immediately" = true ]; then
                echo "Deleting local directory: $dir"
                rm -rf "$dir"
            elif [ "$delete_synced" = true ]; then
                echo "Marking for deletion after all syncs complete"
                synced_dirs+=("$dir")
            else
                echo "Keeping local directory"
            fi
        else
            echo ""
            echo "SYNC FAILED: Exit code $sync_exit_code"
            echo "   This run will NOT be deleted locally."
            failed_count=$((failed_count + 1))
        fi

        rm "$sync_output_file"
        echo ""
    done < "$temp_file"

    # Delete synced directories if requested
    if [ "$delete_synced" = true ] && [ ${#synced_dirs[@]} -gt 0 ]; then
        echo ""
        echo "================================"
        echo "Deleting all successfully synced directories..."
        echo "================================"
        for dir in "${synced_dirs[@]}"; do
            echo "Deleting: $dir"
            rm -rf "$dir"
        done
        echo "Deleted ${#synced_dirs[@]} synced directories."
    fi

    # Cleanup
    rm "$temp_file"

    # Summary
    echo ""
    echo "================================"
    echo "Sync Summary"
    echo "================================"
    echo "Successfully synced: $success_count"
    echo "Failed to sync: $failed_count"
    echo "Total runs processed: $((success_count + failed_count))"
    echo ""
    echo "Sync process completed."
}

# ==============================================================================
# Main
# ==============================================================================

# Check for mode
if [ $# -eq 0 ]; then
    show_help
fi

MODE="$1"
shift

case "$MODE" in
    sync)
        run_sync_mode "$@"
        ;;
    list-projects|find-duplicates|generate-report|cleanup-empty|delete-duplicates)
        # Check if Python manager exists
        if [ ! -f "$WANDB_MANAGER" ]; then
            echo "Error: wandb_manager.py not found at $WANDB_MANAGER"
            exit 1
        fi
        # Run Python manager
        python "$WANDB_MANAGER" "$MODE" "$@"
        ;;
    --help|-h|help)
        show_help
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes (informational -> actions):"
        echo ""
        echo "  INFORMATIONAL:"
        echo "    list-projects    List expected project names"
        echo "    find-duplicates  Find duplicate runs"
        echo "    generate-report  Generate duplicate report"
        echo ""
        echo "  ACTIONS:"
        echo "    sync             Sync offline WandB runs"
        echo "    cleanup-empty    Delete runs with no evaluation metrics"
        echo "    delete-duplicates Delete identical duplicates"
        echo ""
        echo "Use: ./wandb_tools.sh <mode> --help for mode-specific help"
        exit 1
        ;;
esac
