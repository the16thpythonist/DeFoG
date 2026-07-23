#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=09:30:00
#SBATCH --output=chembl_gdpo_sweep_%j.out

# JUPITER: GDPO structural-sanity fine-tune of the ChEMBL foundation model ->
# foundation v2. A 4-GPU KL_COEF sweep (one arm per GPU) mapping the
# reward-vs-diversity tradeoff:
#   GPU0 KL_COEF=0.05   GPU1 KL_COEF=0.1   GPU2 KL_COEF=0.2   GPU3 KL_COEF=0.5
# Reward = binary structural sanity (valid AND connected AND ring-sane), envelope
# derived from the ChEMBL reference. KL-to-frozen-base + validity/uniqueness/
# novelty rails + best-snapshot selection (never ships worse than the input) guard
# against reward-hacking. Each arm reports BEFORE/AFTER on the full extended suite.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

CKPT="ckpts/chembl_foundation_lr3e-4/best_model.ckpt"
REF="data/chembl/chembl_train.smiles"
KLS=(0.05 0.1 0.2 0.5)

# Pre-create the SHARED pycomex namespace dir so the 4 concurrent arms don't race
# on os.mkdir(namespace_dir) in prepare_path() (the FileExistsError that kills arms).
mkdir -p experiments/results/gdpo_sanity

echo "starting ChEMBL GDPO sanity sweep @ $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
      --CKPT_PATH "'$CKPT'" --REFERENCE_SMILES "'$REF'" \
      --REFERENCE_LIMIT 100000 --KL_COEF ${KLS[$i]} --__DEBUG__ False \
      > "gdpo_kl${KLS[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched KL_COEF=${KLS[$i]} on GPU $i (pid $!)"
  sleep 15
done

wait
echo "all GDPO arms finished @ $(date)"
