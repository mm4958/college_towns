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
set -e  # Exit immediately if any command fails

# Directory for logs
LOG_DIR="/home/mm4958/openalex/logs"
mkdir -p "$LOG_DIR"

# Helper function to run a script
run_script() {
    local script_path="$1"
    local log_file="$2"

    echo "---------------------------------------------"
    echo "Running $script_path ..."
    echo "Logging to $log_file"
    echo "---------------------------------------------"

    # Run with timing, redirect stdout & stderr to log
    if ! time python3 "$script_path" >"$log_file" 2>&1; then
        echo "⚠ Script $script_path failed. Check log: $log_file"
        exit 1
    fi

    echo "✓ Completed $script_path"
    echo ""
}

# Run scripts sequentially
run_script "/home/mm4958/openalex/1_allschoolaffiliations_v4.py" "$LOG_DIR/1_allschoolaffiliations_v4.log"
run_script "/home/mm4958/openalex/3_eachschoolyears_v4.py" "$LOG_DIR/3_eachschoolyears_v4.log"
run_script "/home/mm4958/openalex/4_get_hosp.py" "$LOG_DIR/4_get_hosp.log"
run_script "/home/mm4958/openalex/5_get_years_v2.py" "$LOG_DIR/5_get_years_v2.log"
run_script "/home/mm4958/openalex/6_get_final_stats.py" "$LOG_DIR/6_get_final_stats.log"
run_script "/home/mm4958/openalex/7_clean_v3.py" "$LOG_DIR/7_clean_v3.log"
run_script "/home/mm4958/openalex/8_graphs.py" "$LOG_DIR/8_graphs.log"

echo "All scripts completed successfully!"
