#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:30:00
#SBATCH --job-name=logp_rl_r2
#SBATCH --output=slurm-%j.out
#
# ROUND 2 (ratchet) of logP-adapter GDPO RL: 4 seed replicates {42,43,44,45}, all
# starting from the ROUND-1 eta=1 winner (ckpts/logp_adapter_rl_eta1_round1.ckpt).
# Because AdapterGDPOTrainer builds its KL reference from the input adapter, the KL
# re-anchors to the round-1 result -> the policy can push further from the original
# pre-RL adapter than round 1 could (ratchet relaxes the floor), while early-stop
# still guards the trough. Winning config: eta=1, CRN on, K=128/G=8 (16/group),
# KL=0.1, Dr. GRPO. FEWER iters (150, probe every 20) since it peaks ~iter40.
# Tests whether the ratchet gain is robust across seeds.
#
# Submit FROM the repo dir:  sbatch run_adapter_rl_round2_seeds_jupiter.sh
set -u
cd "$SLURM_SUBMIT_DIR"

module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1     # BARE (never pipe -> subshell loses env)
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"                       # APPEND
export PYTHONUNBUFFERED=1

mkdir -p experiments/results/adapter_rl_finetune__zinc    # pre-create namespace (mkdir-race fix)

BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
ADAPTER="ckpts/logp_adapter_rl_eta1_round1.ckpt"          # round-1 winner -> ratchet anchor
SEEDS=(42 43 44 45)

for i in 0 1 2 3; do
  S=${SEEDS[$i]}
  echo "launching arm $i on GPU $i: SEED=$S"
  CUDA_VISIBLE_DEVICES=$i python experiments/adapter_rl_finetune__zinc.py \
      --BASE_CKPT "'$BASE'" --ADAPTER_CKPT "'$ADAPTER'" --PROPERTY "'logp'" \
      --ROLLOUT_ETA 1 --SEED $S \
      --ROLLOUT_SIZE 128 --N_GROUPS 8 --CRN True \
      --KL_COEF 0.1 --EARLY_STOP True \
      --MAX_ITERS 150 --PROBE_EVERY 20 --MAX_TIME_HOURS 1.5 \
      > slurm_r2_seed${S}.out 2>&1 &
  sleep 20
done
wait
echo "all 4 round-2 seed arms finished"
