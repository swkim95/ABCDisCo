#!/bin/bash
# submit_qnn_batches.sh
set -euo pipefail

JOB_SCRIPT="/pscratch/sd/s/sungwon/ABCDisCo/jobs/run_qnn_array.slurm"   # uses -t 8:00:00 and -q regular

# BATCH_SIZE=240
# CONCURRENCY=80

sbatch --array="0-189"  "$JOB_SCRIPT"
sbatch --array="190-379" "$JOB_SCRIPT"
sbatch --array="380-575" "$JOB_SCRIPT"
