#!/bin/bash
# Train one amortized property-guidance net for the ZINC250k base model on the
# local euler GPU (RTX 2060). Usage:  sbatch --job-name=zinc-<prop> slurm_zinc_guidance.sbatch <prop>
# The three adapters are chained (dependency) so only one uses the 6 GB GPU at a time.
#SBATCH --partition=general
#SBATCH --nodelist=euler
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/media/ssd2/Programming/DeFoG/experiments/_zinc_%x_%j.log

set -euo pipefail
PROP="${1:?usage: sbatch slurm_zinc_guidance.sbatch <logp|tpsa|sascore> [epochs]}"
EPOCHS="${2:-60}"   # 249k molecules => far fewer epochs than AqSolDB still over-trains

cd /media/ssd2/Programming/DeFoG
export PYTHONPATH=/media/ssd2/Programming/DeFoG
export CUDA_VISIBLE_DEVICES=0   # euler's single GPU

echo "[slurm] $(date '+%F %T') host=$(hostname) job=${SLURM_JOB_ID:-?} property=${PROP} epochs=${EPOCHS}"

# Full ZINC250k (249k molecules), same recipe as the AqSolDB adapters, batch 16 to
# keep VRAM (~1.5 GB) small enough to coexist with the defog-web sampling worker.
# --atom-decoder pins the vocab to the ZINC checkpoint's node-class order.
.venv/bin/python experiments/train_property_guidance.py \
  --property "${PROP}" \
  --ckpt /media/ssd2/Programming/defog-web/models/zinc/zinc_uncond_4e-4_best_model.ckpt \
  --data data/zinc_250k_rdkit.csv \
  --atom-decoder C,N,O,F,P,S,Cl,Br,I \
  --outdir experiments/_zinc_${PROP}_guidance \
  --epochs "${EPOCHS}" --batch-size 16

echo "[slurm] $(date '+%F %T') property=${PROP} DONE"
