#!/bin/bash
#SBATCH --job-name=openalex
#SBATCH --output=/home/mm4958/openalex/logs/openalex_%j.out
#SBATCH --error=/home/mm4958/openalex/logs/openalex_%j.err
#SBATCH -t 3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p sched_mit_sloan_batch

# Load modules if needed
# module load python/3.10
source /orcd/software/core/001/centos7/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate ct_env
# Make sure errors stop the job
set -e

# Directory for logs
LOG_DIR="/home/mm4958/openalex/logs"
mkdir -p "$LOG_DIR"

run_script() {
    local script_path="$1"
    local log_file="$2"

    echo "---------------------------------------------"
    echo "Running $script_path ..."
    echo "Logging to $log_file"
    echo "---------------------------------------------"

    if ! time python3 "$script_path" >"$log_file" 2>&1; then
        echo "⚠ Script $script_path failed. Check log: $log_file"
        exit 1
    fi

    echo "✓ Completed $script_path"
    echo ""
}

run_script "/home/mm4958/openalex/check.py" "$LOG_DIR/check.log"

echo "All scripts completed successfully!"
