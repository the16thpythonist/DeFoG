#!/bin/bash
#SBATCH --job-name=gdpo_sanity_arm
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=gdpo_sanity_%x_%j.out

# Single-arm, 1-GPU launcher for the structural-sanity sweep. Use this instead of the
# 4-GPU run_gdpo_sanity_zinc_sweep_kcist.sh when no full 4-GPU node is free: four
# independent 1-GPU jobs BACKFILL onto free GPUs across mix nodes and start immediately.
# PRE-BUILD the reference cache once before submitting (see below) so the arms don't
# race on the cache write.
#
# Usage (label kl iters positive_only lr):
#   sbatch --job-name=sane_A run_gdpo_sanity_zinc_1gpu_kcist.sh A_kl0.4 0.4 40 False 2e-5
#   sbatch --job-name=sane_B run_gdpo_sanity_zinc_1gpu_kcist.sh B_kl0.8 0.8 40 False 2e-5
#   sbatch --job-name=sane_C run_gdpo_sanity_zinc_1gpu_kcist.sh C_raft  0.3 40 True  2e-5
#   sbatch --job-name=sane_D run_gdpo_sanity_zinc_1gpu_kcist.sh D_lowlr 0.3 60 False 1e-5

LABEL=${1:?need label}; KL=${2:?need kl}; IT=${3:?need iters}; PO=${4:?need positive_only}; LR=${5:?need lr}

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p experiments/results/gdpo_sanity
CK="$HOME/zinc_round1.ckpt"; REF="data/zinc_250k_rdkit.csv"
echo "SANITY ARM $LABEL (kl=$KL iters=$IT positive_only=$PO lr=$LR) | $(date)"
ls -la "$CK" || { echo "MISSING $CK"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
python -u experiments/gdpo_sanity.py \
    --__DEBUG__ False --CKPT_PATH "'$CK'" --REFERENCE_SMILES "'$REF'" \
    --ENVELOPE_QUANTILE 1.0 --RING_MIN_COUNT 50 --REQUIRE_CONNECTED True \
    --KL_COEF $KL --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
    --POSITIVE_ONLY $PO --LAMBDA_EDGE 1 --ROUNDS 1 --ITERATIONS $IT \
    --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --TIME_DISTORTION "'polydec'" \
    --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 --ETA 0.0 --LR $LR --EMA_DECAY 0.9 \
    --CKPT_EVERY 10 --SELECT_BEST True --SELECT_INCLUDE_INPUT True \
    --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
    --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --COMPUTE_FCD True --FCD_REF_SAMPLES 10000 --SEED 1

echo "arm $LABEL done at $(date)"
