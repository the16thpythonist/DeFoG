#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --job-name=logp_rl_eta
#SBATCH --output=slurm-%j.out
#
# 4-GPU ROLLOUT_ETA sweep for RL-fine-tuning the ZINC logP adapter (GDPO), with the
# new rollout machinery: 16 rollouts/group (K=128, G=8), common-random-numbers within
# each target group (CRN), Dr. GRPO (advantage_mode="mean"), and validity-gated
# early-stop. One arm per GPU; eval fixed at the deployment policy (w=1, eta=5, 500
# steps). Under CRN, ROLLOUT_ETA is the sole within-group diversity source -- this
# sweep asks which rollout exploration temperature trains the best-steering adapter.
#
# Submit FROM the repo dir:  sbatch run_adapter_rl_eta_sweep_jupiter.sh
set -u
cd "$SLURM_SUBMIT_DIR"

module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1     # BARE (never pipe -> subshell loses env)
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"                       # APPEND (module torch/numpy must stay importable)
export PYTHONUNBUFFERED=1

# Fix Bug A (pycomex concurrent-mkdir race): pre-create the shared namespace dir so
# the 4 arms don't TOCTOU-race on os.mkdir(namespace) at startup.
mkdir -p experiments/results/adapter_rl_finetune__zinc

# Both ckpts must exist under ckpts/ on JUPITER (gitignored -> transferred via ctun).
BASE="ckpts/zinc_uncond_4e-4_connectivity.ckpt"
ADAPTER="ckpts/logp_adapter_preRL_dBe2.ckpt"
ETAS=(1 5 15 40)

for i in 0 1 2 3; do
  ETA=${ETAS[$i]}
  echo "launching arm $i on GPU $i: ROLLOUT_ETA=$ETA"
  CUDA_VISIBLE_DEVICES=$i python experiments/adapter_rl_finetune__zinc.py \
      --BASE_CKPT "'$BASE'" --ADAPTER_CKPT "'$ADAPTER'" --PROPERTY "'logp'" \
      --ROLLOUT_ETA $ETA \
      --ROLLOUT_SIZE 128 --N_GROUPS 8 --CRN True \
      --KL_COEF 0.1 --EARLY_STOP True --MAX_TIME_HOURS 3.0 \
      > slurm_eta${ETA}.out 2>&1 &
  sleep 20                                                 # stagger so archive-folder creation doesn't collide
done
wait
echo "all 4 eta arms finished"
