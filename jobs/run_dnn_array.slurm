#!/bin/bash

#SBATCH -J abcd-dnn
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --array=0-63%2
#SBATCH -L scratch,cfs
#SBATCH -D /pscratch/sd/s/sungwon/ABCDisCo
#SBATCH -o /pscratch/sd/s/sungwon/ABCDisCo/slurm_log/dnn/output/%x-%A_%a.out
#SBATCH -e /pscratch/sd/s/sungwon/ABCDisCo/slurm_log/dnn/error/%x-%A_%a.err

# If your default account is not set, uncomment and set your project account:
#SBATCH --account=m4138

set -euo pipefail

source ~/.bashrc
setup

echo "[$(date)] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_NODELIST}"
echo "Workdir: ${PWD}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# Hyperparameter grids
ALPHAS=(50 100 150 200)
LRS=(1e-4 5e-4 1e-3 3e-3)
SMEARS=(0 25 50 75)

task_id=${SLURM_ARRAY_TASK_ID}
n_lr=${#LRS[@]}
n_sm=${#SMEARS[@]}

alpha_idx=$(( task_id / (n_lr * n_sm) ))
rem=$(( task_id % (n_lr * n_sm) ))
lr_idx=$(( rem / n_sm ))
smear_idx=$(( rem % n_sm ))

ALPHA=${ALPHAS[$alpha_idx]}
LR=${LRS[$lr_idx]}
SMEAR=${SMEARS[$smear_idx]}

RUN_LABEL="dnn_a${ALPHA}_lr${LR}_s${SMEAR}"
OUT_ROOT="runs/dnn"
mkdir -p "${OUT_ROOT}"

echo "Launching: python -u script/train_abcd_single.py --alpha ${ALPHA} --lr ${LR} --smear ${SMEAR} --gpunum \"\${CUDA_VISIBLE_DEVICES}\" --run_label ${RUN_LABEL} --output_root ${OUT_ROOT}"
srun --cpu-bind=cores --gpus=1 --gpu-bind=single:1 \
  --output "/pscratch/sd/s/sungwon/ABCDisCo/slurm_log/dnn/output/${RUN_LABEL}_%A_%a.out" \
  --error  "/pscratch/sd/s/sungwon/ABCDisCo/slurm_log/dnn/error/${RUN_LABEL}_%A_%a.err" \
  -u bash -lc \
  "python -u script/train_abcd_single.py \
    --alpha ${ALPHA} \
    --lr ${LR} \
    --smear ${SMEAR} \
    --gpunum \"\${CUDA_VISIBLE_DEVICES}\" \
    --run_label ${RUN_LABEL} \
    --output_root ${OUT_ROOT} \
    --test --epochs 1"


