#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:30:00
#SBATCH --job-name=logp_rl_lrfix
#SBATCH --output=slurm-%j.out
#
# DEFINITIVE test of the RL fixes at a CLEAN K=128 gradient: does GDPO actually
# TIGHTEN the logP adapter? Fixes under test: LR 1e-4 (was 1e-5, too weak), fast-EMA
# 0.9 (was 0.999, lagged near-original), PAIRED eval (fixed EVAL_SEED so pre/post use
# the same draws -> real delta, not sampling noise), deploy-step probe (early-stop
# selects a snapshot best at the 500-step deployment), and a weight-diff sanity check.
# 4 seeds {42-45} from the ORIGINAL pre-RL adapter, short (~120 iters, ~1.5h RL).
# READ per arm: PAIRED pre->post logP MAE (low+high) + deployed_weight_diff (>>1e-5).
#
# Submit FROM the repo dir:  sbatch run_adapter_rl_lrfix_seeds_jupiter.sh
set -u
cd "$SLURM_SUBMIT_DIR"

module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1     # BARE
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"                       # APPEND
export PYTHONUNBUFFERED=1

mkdir -p experiments/results/adapter_rl_finetune__zinc    # pre-create namespace (mkdir-race fix)

BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
ADAPTER="ckpts/logp_adapter_preRL_dBe2.ckpt"              # ORIGINAL pre-RL adapter
SEEDS=(42 43 44 45)

for i in 0 1 2 3; do
  S=${SEEDS[$i]}
  echo "launching arm $i on GPU $i: SEED=$S"
  CUDA_VISIBLE_DEVICES=$i python experiments/adapter_rl_finetune__zinc.py \
      --BASE_CKPT "'$BASE'" --ADAPTER_CKPT "'$ADAPTER'" --PROPERTY "'logp'" \
      --LR 1e-4 --EMA_DECAY 0.9 --ROLLOUT_ETA 1 --SEED $S \
      --ROLLOUT_SIZE 128 --N_GROUPS 8 --CRN True --KL_COEF 0.1 \
      --MAX_ITERS 120 --MAX_TIME_HOURS 1.5 --EARLY_STOP True \
      --PROBE_EVERY 30 --PROBE_N_PER 32 \
      > slurm_lrfix_seed${S}.out 2>&1 &
  sleep 20
done
wait
echo "all 4 lr-fix seed arms finished"
