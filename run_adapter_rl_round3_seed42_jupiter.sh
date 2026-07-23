#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:30:00
#SBATCH --job-name=logp_rl_r3
#SBATCH --output=slurm-%j.out
#
# RATCHET round-3 from the round-2 seed42 winner
# (ckpts/logp_adapter_rl_seed42_round2.ckpt), KL re-anchored to it. Does the ratchet
# keep compounding past round 2 (low 0.55 / high 0.47, low-validity 98%)? Same config
# (LR 1e-4, fast-EMA 0.9, eta=1, CRN, K=128/G=8, KL=0.1, paired eval, deploy-step
# probe, weight-diff). 4 seeds {42-45}. Report PAIRED pre->post MAE + validity/target.
#
# Submit FROM the repo dir:  sbatch run_adapter_rl_round3_seed42_jupiter.sh
set -u
cd "$SLURM_SUBMIT_DIR"

module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1     # BARE
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"                       # APPEND
export PYTHONUNBUFFERED=1

mkdir -p experiments/results/adapter_rl_finetune__zinc    # pre-create namespace (mkdir-race fix)

BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
ADAPTER="ckpts/logp_adapter_rl_seed42_round2.ckpt"        # ROUND-2 seed42 winner -> ratchet anchor
SEEDS=(42 43 44 45)

for i in 0 1 2 3; do
  S=${SEEDS[$i]}
  echo "launching arm $i on GPU $i: SEED=$S"
  CUDA_VISIBLE_DEVICES=$i python experiments/adapter_rl_finetune__zinc.py \
      --BASE_CKPT "'$BASE'" --ADAPTER_CKPT "'$ADAPTER'" --PROPERTY "'logp'" \
      --LR 1e-4 --EMA_DECAY 0.9 --ROLLOUT_ETA 1 --SEED $S \
      --ROLLOUT_SIZE 128 --N_GROUPS 8 --CRN True --KL_COEF 0.1 \
      --MAX_ITERS 120 --MAX_TIME_HOURS 1.5 --EARLY_STOP True \
      --PROBE_EVERY 20 --PROBE_N_PER 32 \
      > slurm_r3_seed${S}.out 2>&1 &
  sleep 20
done
wait
echo "all 4 round-3 seed arms finished"
