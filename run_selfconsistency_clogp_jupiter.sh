#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=07:00:00
#SBATCH --output=sc_clogp_%j.out

# JUPITER: PROTOTYPE test of the property-head + self-consistency loss on a NEW ClogP
# interior adapter. 4 arms sweep the self-consistency loss weight LAMBDA_SC:
#   GPU0 lsc=0.0 (ablation: grounded head, no adapter coupling)
#   GPU1 lsc=0.3   GPU2 lsc=1.0   GPU3 lsc=3.0
# Each: trains adapter + head (base frozen), then reports H-vs-RDKit MAE + steering MAE@w1.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=32

BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
LSC=(0.0 0.3 1.0 3.0)

echo "starting self-consistency ClogP sweep at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Pre-create the SHARED pycomex namespace dir so the 4 concurrent arms don't race
# on os.mkdir(namespace_dir) in prepare_path() (the FileExistsError that killed the
# lsc=0.3/1.0 arms). A pre-existing dir => pycomex skips the mkdir => no race.
mkdir -p experiments/results/adapter_selfconsistency__zinc

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -u experiments/adapter_selfconsistency__zinc.py \
      --PROPERTY "'logp'" --LAMBDA_SC ${LSC[$i]} \
      --BASE_CKPT "'$BASE'" \
      > "sc_clogp_lsc${LSC[$i]}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched lambda_sc=${LSC[$i]} on GPU $i (pid $!)"
  sleep 15
done

wait
echo "all self-consistency arms finished at $(date)"
