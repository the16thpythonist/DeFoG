#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=chembl_gdpo_smoke_%j.out

# JUPITER: SMOKE the GDPO structural-sanity fine-tune wiring on the ChEMBL
# foundation model (testing mode: 4 iters, tiny rollout/eval). Validates that the
# 12-element schema loads, the ChEMBL reference parses + builds the sanity
# envelope, one GDPO iteration runs, and eval runs -- before the real 4-GPU sweep.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "GDPO ChEMBL smoke @ $(date)"
python -u experiments/gdpo_sanity.py --__TESTING__ True \
    --CKPT_PATH "'ckpts/chembl_foundation_lr3e-4/best_model.ckpt'" \
    --REFERENCE_SMILES "'data/chembl/chembl_train.smiles'"
echo "smoke done @ $(date)"
