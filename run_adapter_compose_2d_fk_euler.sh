#!/bin/bash
#SBATCH --job-name=compose2d_fk
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=compose_2d_fk_%j.out

# Feynman-Kac refinement ON TOP OF the adapter composition, with the DEPLOYED
# head-RL adapters (what defog-web serves) rather than the pre-RL pair used for the
# figure reproduction -- so this differs from job 3732 in two ways at once, adapters
# AND FK.
#
# Reward = the trained property heads (learned surrogate), combined by
# JointLearnedEnergy with per-property variance normalization. Note the plot's axes
# are RDKit logP/TPSA, so FK optimizes the surrogate while the figure scores truth.
#
# K=100 is the SMC pool and the GPU batch at once. Measured on this RTX 2060
# (job 3734, worst-case 38-node graphs): K=100 -> 2.66 GiB, K=160 -> 4.19 GiB,
# K=200 -> OOM. 100 x 2 batches gives exactly 200/combo with uniform pools, so every
# molecule sees the same selection pressure and the count matches jobs 3731/3732.
set -euo pipefail

cd /media/ssd2/Programming/DeFoG
export PYTHONUNBUFFERED=1
export PYTHONPATH=/media/ssd2/Programming/DeFoG
export CUDA_VISIBLE_DEVICES=0

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true

.venv/bin/python -u experiments/adapter_compose_2d__zinc.py \
    --BASE_CKPT "'/home/jonas/Downloads/zinc_uncond_4e-4_connectivity.ckpt'" \
    --LOGP_CKPT "'/media/ssd2/Programming/DeFoG/ckpts/logp_head_rl_final.ckpt'" \
    --TPSA_CKPT "'/media/ssd2/Programming/DeFoG/ckpts/tpsa_head_rl_final.ckpt'" \
    --USE_FK "True" \
    --FK_PARTICLES "100" \
    --FK_BETA "2.5" \
    --FK_WARMUP_FRAC "0.5" \
    --SEED "43" \
    --N_PER_COMBO "200" \
    --EVAL_STEPS "250" \
    --BACKGROUND "'contour'"

echo "finished=$(date -Is)"
