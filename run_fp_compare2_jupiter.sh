#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=fp_compare2_%j.out

# Matched-width head-to-head: the SHIPPED v2.0.0 (binary/512, no interior)
# against the new best (counts/1024 + interior_ff). Needed before shipping,
# because the two live at different fingerprint widths and Tanimoto is not
# comparable across widths -- the mistake that invalidated the first v2-vs-v3
# comparison. Chaining "v2 < v3 (matched) and v3 < ff (in-job)" is an inference,
# not a measurement; this measures it.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

V2=experiments/results/adapter_fingerprint__zinc/06_08_2026__16_20__wbxr/fp_adapter
FF=experiments/results/adapter_fingerprint__zinc/08_08_2026__13_03__pkM7/fp_adapter
for p in "$V2" "$FF"; do [ -f "${p}.ckpt" ] || { echo "ERROR: ${p}.ckpt missing"; exit 1; }; done

python -u scripts/compare_fp_adapters.py \
  --base ckpts/zinc_rl2_seed42/best_model \
  --v2 "$V2" --v2-bits 512 \
  --v3 "$FF" --v3-bits 1024 --v3-counts \
  --n-per-target 64 --n-baseline 256 --steps 500 --eta 25 \
  --out fp_compare2_${SLURM_JOB_ID}.json
