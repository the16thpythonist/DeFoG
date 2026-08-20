#!/bin/bash
#SBATCH --job-name=compose2d_contour
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=compose_2d_contour_%j.out

# Local (euler) counterpart of run_adapter_compose_2d_jupiter.sh: same frozen base
# + same two adapters (md5-identical to the JUPITER run), fresh sampling at SEED=43,
# and the KDE-contour background instead of hexbin.
#
# euler's RTX 2060 (6 GB) is NOT managed by SLURM (GRES=null on both nodes), so the
# GPU is taken opportunistically rather than requested -- the CPU/mem request is
# deliberately small so this schedules alongside whatever else holds the node.
# EVAL_CHUNK drops 40 -> 16 so the 9-layer/256-hidden model fits in 6 GB next to the
# desktop session; product-mode composition runs base + 2 branches per step.
set -euo pipefail

cd /media/ssd2/Programming/DeFoG
export PYTHONUNBUFFERED=1
export PYTHONPATH=/media/ssd2/Programming/DeFoG
export CUDA_VISIBLE_DEVICES=0

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

.venv/bin/python -u experiments/adapter_compose_2d__zinc.py \
    --BASE_CKPT "'/home/jonas/Downloads/zinc_uncond_4e-4_connectivity.ckpt'" \
    --LOGP_CKPT "'/media/ssd2/Programming/DeFoG/ckpts/logp_adapter_preRL_dBe2.ckpt'" \
    --TPSA_CKPT "'/media/ssd2/Programming/DeFoG/ckpts/tpsa_adapter_preRL_jFD3.ckpt'" \
    --SEED "43" \
    --N_PER_COMBO "200" \
    --EVAL_STEPS "250" \
    --EVAL_CHUNK "16" \
    --BACKGROUND "'contour'"

echo "finished=$(date -Is)"
