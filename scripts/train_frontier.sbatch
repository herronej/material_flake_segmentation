#!/bin/bash
#SBATCH -A <PROJECT_ID>
#SBATCH -J flakeseg
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH -c 7

# Frontier: one task per GCD, 8 GCDs per node (4 MI250X).
# Adjust -N for multi-node; the launcher below handles rendezvous either way.

set -euo pipefail

module load PrgEnv-gnu
# Pinned, not bare `rocm`: the unversioned module resolves to the site default
# (6.2.4 as of this writing), while the wheel is torch==2.8.0+rocm6.4. A
# mismatched module puts its own ROCm libs ahead of the ones bundled in the
# wheel and you get loader errors that look like a broken install. Keep this
# in step with whatever `pip install --index-url .../rocm6.4` provided.
module load rocm/6.4.2
module load craype-accel-amd-gfx90a

source "${FLAKESEG_VENV:-$HOME/envs/flakeseg}/bin/activate"

# NOTE: MIOpen cache paths are set per rank inside the srun wrapper below, not
# here. In the batch script SLURM_PROCID is 0 (or unset), so setting them at
# this scope hands all 8 ranks on the node the same node-local SQLite cache and
# they serialize on its lock while autotuning kernels.

# RCCL over Slingshot. Drop these if your site module already sets them.
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATOMIC_TXN=1

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CONFIG="${1:-configs/graphene.yaml}"
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

mkdir -p logs

srun --unbuffered bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$SLURM_NTASKS
  export MIOPEN_USER_DB_PATH="/tmp/miopen-$SLURM_JOB_ID-$SLURM_PROCID"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  mkdir -p "$MIOPEN_USER_DB_PATH"
  exec python -m flakeseg.train --config '"$CONFIG"'
'
