#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=fp_compare_%j.out

# Head-to-head v2 (binary/512) vs v3 (counts/1024) with the METRIC width held
# fixed. The original comparison was invalid: I kept the metric binary but let
# it be computed at each adapter's own width, and Tanimoto at 1024 bits is
# systematically lower than at 512 (fewer collisions -> less spurious overlap),
# which is why the two runs report different baselines for the same base model.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

V2=experiments/results/adapter_fingerprint__zinc/06_08_2026__16_20__wbxr/fp_adapter
V3=experiments/results/adapter_fingerprint__zinc/07_08_2026__11_01__Jzid/fp_adapter

for p in "$V2" "$V3"; do
  [ -f "${p}.ckpt" ] || { echo "ERROR: ${p}.ckpt missing"; exit 1; }
done

python -u scripts/compare_fp_adapters.py \
  --base ckpts/zinc_rl2_seed42/best_model \
  --v2 "$V2" --v2-bits 512 \
  --v3 "$V3" --v3-bits 1024 --v3-counts \
  --n-per-target 64 --n-baseline 256 --steps 500 --eta 25 \
  --out fp_compare_${SLURM_JOB_ID}.json
