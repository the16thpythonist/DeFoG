#!/bin/bash
#SBATCH --job-name=fk_fp_zinc
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=fk_fp_zinc_%j.out

# Single-GPU: Feynman-Kac (SMC) fingerprint steering combined WITH the trained
# latent guidance adapter. Sampling-only (loads the adapter from job 26407) so it
# starts immediately. Ablation: baseline vs guidance vs fk vs fk+guidance.

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

mkdir -p experiments/results/fk_fingerprint_steer__zinc

echo "starting FK fingerprint-steer run at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# pycomex eval()s CLI values -> string params need to be Python string literals.
python -u experiments/fk_fingerprint_steer__zinc.py \
    --BASE_CKPT "'/home/tm4030/Programming/DeFoG/ckpts/zinc_uncond_4e-4_connectivity.ckpt'"

echo "run finished at $(date)"
