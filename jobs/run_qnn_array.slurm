#!/bin/bash

#SBATCH -J abcd-qnn
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --array=0-575%2
#SBATCH -L scratch,cfs
#SBATCH -D $SCRATCH/ABCDisCo
#SBATCH -o $SCRATCH/ABCDisCo/slurm_log/qnn/output/%x-%A_%a.out
#SBATCH -e $SCRATCH/ABCDisCo/slurm_log/qnn/error/%x-%A_%a.err

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
NQ=(6 8 11)
DEPTHS=(4 6 8)

task_id=${SLURM_ARRAY_TASK_ID}
n_lr=${#LRS[@]}
n_sm=${#SMEARS[@]}
n_nq=${#NQ[@]}
n_d=${#DEPTHS[@]}

alpha_idx=$(( task_id / (n_lr * n_sm * n_nq * n_d) ))
rem=$(( task_id % (n_lr * n_sm * n_nq * n_d) ))
lr_idx=$(( rem / (n_sm * n_nq * n_d) ))
rem=$(( rem % (n_sm * n_nq * n_d) ))
smear_idx=$(( rem / (n_nq * n_d) ))
rem=$(( rem % (n_nq * n_d) ))
nq_idx=$(( rem / n_d ))
depth_idx=$(( rem % n_d ))

ALPHA=${ALPHAS[$alpha_idx]}
LR=${LRS[$lr_idx]}
SMEAR=${SMEARS[$smear_idx]}
N_QUBIT=${NQ[$nq_idx]}
VQC_DEPTH=${DEPTHS[$depth_idx]}

RUN_LABEL="qnn_a${ALPHA}_lr${LR}_s${SMEAR}_nq${N_QUBIT}_d${VQC_DEPTH}"
OUT_ROOT="runs/qnn"
mkdir -p "${OUT_ROOT}"
mkdir -p "$SCRATCH/ABCDisCo/slurm_log/qnn/output" "$SCRATCH/ABCDisCo/slurm_log/qnn/error"

echo "Launching: python -u QNN_script/train_QNN_abcd_single.py --alpha ${ALPHA} --lr ${LR} --smear ${SMEAR} --qnn-n-qubits ${N_QUBIT} --qnn-vqc-depth ${VQC_DEPTH} --gpunum \"\${CUDA_VISIBLE_DEVICES}\" --run_label ${RUN_LABEL} --output_root ${OUT_ROOT}"
srun --cpu-bind=cores --gpus=1 --gpu-bind=single:1 \
  --output "$SCRATCH/ABCDisCo/slurm_log/qnn/output/${RUN_LABEL}_%A_%a.out" \
  --error  "$SCRATCH/ABCDisCo/slurm_log/qnn/error/${RUN_LABEL}_%A_%a.err" \
  -u bash -lc \
  "python -u QNN_script/train_QNN_abcd_single.py \
    --alpha ${ALPHA} \
    --lr ${LR} \
    --smear ${SMEAR} \
    --qnn-n-qubits ${N_QUBIT} \
    --qnn-vqc-depth ${VQC_DEPTH} \
    --gpunum \"\${CUDA_VISIBLE_DEVICES}\" \
    --run_label ${RUN_LABEL} \
    --output_root ${OUT_ROOT} \
    --test --epochs 1"


