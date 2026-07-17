#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:00:00
#SBATCH --output=compose_2d_%j.out

# JUPITER: 2D adapter-STACKING validation. Compose the trained logP + TPSA
# adapters (product-of-experts, w=1) over the frozen base, steer all 4 high/low
# combos, sample from a property-CONDITIONAL size distribution, and plot over the
# 2D logP x TPSA dataset density. Sampling-only (uses 1 GPU of the node).

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

mkdir -p experiments/results/adapter_compose_2d__zinc
echo "starting 2D compose validation at $(date)"

# best-per-property adapters from job 966737 (logp@2e-4, tpsa@4e-4). pycomex
# eval()s CLI values -> string paths need the extra single quotes.
python -u experiments/adapter_compose_2d__zinc.py \
    --BASE_CKPT "'ckpts/zinc_uncond_4e-4_connectivity.ckpt'" \
    --LOGP_CKPT "'experiments/results/adapter_training__zinc/17_07_2026__19_32__dBe2/logp_adapter.ckpt'" \
    --TPSA_CKPT "'experiments/results/adapter_training__zinc/17_07_2026__19_32__jFD3/tpsa_adapter.ckpt'"

echo "compose finished at $(date)"
