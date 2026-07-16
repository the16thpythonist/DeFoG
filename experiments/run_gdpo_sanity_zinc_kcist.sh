#!/bin/bash
#SBATCH --job-name=gdpo_sanity_zinc
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=gdpo_sanity_zinc_%j.out

# STRUCTURAL-SANITY fine-tune, ZINC-first prototype (option A). STACKS on the ZINC
# connectivity-improved checkpoint (~/zinc_round1.ckpt): the binary reward is
# valid & connected & sane, where "sane" = every ring size in {3..8} and #rings /
# largest fused-or-spiro ring-system / spiro / bridgehead within the ZINC training
# envelope (moderate: ENVELOPE_QUANTILE=1.0, RING_MIN_COUNT=50 -> rings {3-8},
# max_rings=9, ring_system=7, spiro=3, bridge=8; ~0.02% of real ZINC rejected).
# Best-snapshot is input-inclusive + validity/uniqueness-gated, so it CANNOT ship a
# model worse than what it started from. Headline: structural-violation rate down;
# fidelity cross-checks: ring-histogram TV + FCD down; guardrails: valid/uniq/novel.
#   config: kl=0.2, GRPO, reduction=sum, lambda_edge=1, 100 iters, lr=2e-5, ema=0.9,
#           eta=0, SAMPLE_STEPS=100, best-snapshot, eval@500 + FCD.
#
# PREREQUISITES on KCIST: git pull; ~/.local/bin/uv pip install fcd_torch (into .venv);
#   ~/zinc_round1.ckpt and data/zinc_250k_rdkit.csv present.
#   git pull && sbatch experiments/run_gdpo_sanity_zinc_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p experiments/results/gdpo_sanity   # pycomex prepare_path race guard

CK="$HOME/zinc_round1.ckpt"
echo "GDPO SANITY (ZINC, moderate envelope, kl=0.2, eval@500 + FCD) | $(date)"
echo "ckpt: $CK"; ls -la "$CK" || { echo "MISSING $CK"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
python -u experiments/gdpo_sanity.py \
    --__DEBUG__ False \
    --CKPT_PATH "'$CK'" \
    --REFERENCE_SMILES "'data/zinc_250k_rdkit.csv'" \
    --ENVELOPE_QUANTILE 1.0 --RING_MIN_COUNT 50 --REQUIRE_CONNECTED True \
    --KL_COEF 0.2 --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
    --LAMBDA_EDGE 1 --ROUNDS 1 --ITERATIONS 100 --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 \
    --TIME_DISTORTION "'polydec'" --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 --ETA 0.0 \
    --LR 2e-5 --EMA_DECAY 0.9 \
    --CKPT_EVERY 20 --SELECT_BEST True --SELECT_INCLUDE_INPUT True \
    --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
    --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --COMPUTE_FCD True --FCD_REF_SAMPLES 10000 \
    --SEED 1

echo "done at $(date)"
# Results:  grep -E 'SUMMARY|envelope:|BEFORE:' gdpo_sanity_zinc_${SLURM_JOB_ID}.out
