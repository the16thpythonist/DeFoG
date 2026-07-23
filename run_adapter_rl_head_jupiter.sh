#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:30:00
#SBATCH --job-name=rl_head
#SBATCH --output=slurm-%j.out
#
# Head-reward RL round for ONE property (arg $1: logp|tpsa|qed|sascore): 4 seeds {42-45}
# from that property's INTERIOR SC adapter, REWARD = the learned PropertyHead (the probe +
# best-seed selection use the head; EVAL reports RDKit truth too, to validate). Validity is
# monitor-only (gate disabled via VALIDITY_FLOOR_MARGIN=1.0). Round-2 re-runs this with the
# adapter overridden ($2) to the round-1 winner. Submit ONE node PER property:
#   sbatch run_adapter_rl_head_jupiter.sh qed
#   sbatch run_adapter_rl_head_jupiter.sh qed ckpts/qed_head_rl_round1.ckpt   # round-2
set -u
cd "$SLURM_SUBMIT_DIR"
PROP=${1:?usage: sbatch run_adapter_rl_head_jupiter.sh <logp|tpsa|qed|sascore> [adapter_ckpt]}
BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
ADAPTER=${2:-ckpts/${PROP}_adapter_sc.ckpt}    # default = the SC bundle adapter (round-1)
HEAD="ckpts/${PROP}_head.ckpt"

module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1     # BARE
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"                       # APPEND
export PYTHONUNBUFFERED=1
mkdir -p experiments/results/adapter_rl_finetune__zinc    # pre-create namespace (mkdir-race fix)

SEEDS=(42 43 44 45)
for i in 0 1 2 3; do
  S=${SEEDS[$i]}
  echo "launching arm $i GPU $i: PROP=$PROP SEED=$S ADAPTER=$ADAPTER"
  CUDA_VISIBLE_DEVICES=$i python experiments/adapter_rl_finetune__zinc.py \
      --BASE_CKPT "'$BASE'" --ADAPTER_CKPT "'$ADAPTER'" --HEAD_CKPT "'$HEAD'" \
      --PROPERTY "'$PROP'" --REWARD_SOURCE "'head'" \
      --LR 1e-4 --EMA_DECAY 0.9 --ROLLOUT_ETA 1 --SEED $S \
      --ROLLOUT_SIZE 128 --N_GROUPS 8 --CRN True --KL_COEF 0.1 \
      --MAX_ITERS 120 --MAX_TIME_HOURS 1.5 --EARLY_STOP True --VALIDITY_FLOOR_MARGIN 1.0 \
      --PROBE_EVERY 30 --PROBE_N_PER 32 \
      > "slurm_head_${PROP}_seed${S}.out" 2>&1 &
  sleep 20
done
wait
echo "all 4 ${PROP} head-RL seed arms finished"
