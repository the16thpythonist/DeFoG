#!/bin/bash
#SBATCH --job-name=fp_guid_zinc
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=fp_guid_zinc_%j.out

# Single-GPU: latent (128-bit Morgan fingerprint) guidance adapter on ZINC 250k.
# Frozen base = connectivity-improved ZINC uncond model. Trains an amortized
# LatentGuidanceModule (Tanimoto reward, positive-biased pairing from top-K
# neighbors), then evaluates guided-vs-unconditional Tanimoto to held-out targets.

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

# Pre-create the pycomex namespace dir (dodges the prepare_path TOCTOU race).
mkdir -p experiments/results/fingerprint_guidance__zinc

echo "starting fingerprint-guidance ZINC run at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# pycomex arg parser eval()s CLI values, so string params must be a Python string
# literal -> the extra single quotes are intentional (bash strips the doubles).
python -u experiments/fingerprint_guidance__zinc.py \
    --BASE_CKPT "'/home/tm4030/Programming/DeFoG/ckpts/zinc_uncond_4e-4_connectivity.ckpt'"

echo "run finished at $(date)"
