#!/bin/bash
#SBATCH --job-name=gdpo_squeeze3
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_squeeze3_%j.out

# GDPO connectivity -- apply the KNOWN-GOOD single-round config to the three
# unconditional models (AqSolDB / ZINC / GuacaMol), one per GPU, to squeeze the
# disconnected fraction down a bit without breaking validity.
#
# Proven recipe (single round from pretrained, fixed strong anchor -> stable,
# validity preserved; this is what took ZINC 11.9%->6.8% with validity 87%->92%):
#   ROUNDS=1  ITERATIONS=100  GRPO  KL_ANCHOR=fixed  KL_COEF=0.3  REDUCTION=sum
#   ETA=0 (matched eval)  LR=2e-5  EMA_DECAY=0.9
# GuacaMol has max_nodes=88 (vs 30/38) so its K/minibatch are trimmed for memory.
# CKPT_EVERY=20 keeps pre-collapse snapshots in case any model overtrains.
#
# PREREQUISITE (already uploaded): ~/{aqsoldb,zinc_uncond,guacamol}_pretrained.ckpt.
#   git pull  &&  sbatch experiments/run_gdpo_squeeze_3datasets_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# pycomex prepare_path race guard
mkdir -p experiments/results/gdpo_connectivity

NAMES=(aqsoldb zinc guacamol)
CKPTS=("$HOME/aqsoldb_pretrained.ckpt" "$HOME/zinc_uncond_pretrained.ckpt" "$HOME/guacamol_pretrained.ckpt")
KS=(128 128 64)      # GuacaMol (max_nodes 88) gets a smaller batch
MBS=(16 16 8)
echo "GDPO squeeze on 3 uncond models (${NAMES[*]}) | 1 round x 100 iters | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2; do
  NM=${NAMES[$i]}; CK=${CKPTS[$i]}; K=${KS[$i]}; MB=${MBS[$i]}
  if [ ! -f "$CK" ]; then echo "SKIP $NM: $CK not found"; continue; fi
  # pycomex evals --PARAM values -> quote STRING literals.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CK'" \
      --ROUNDS 1 --ITERATIONS 100 \
      --KL_COEF 0.3 --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --ROLLOUT_SIZE "$K" --MINIBATCH_SIZE "$MB" \
      --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --ETA 0.0 \
      --LR 2e-5 --EMA_DECAY 0.9 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 100 --CKPT_EVERY 20 --SEED 0 \
      > "gdpo_squeeze_${NM}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched $NM (K=$K mb=$MB) on GPU $i (pid $!)"
  sleep 8
done

wait
echo "all arms finished at $(date)"
# Results:  grep -H 'SUMMARY' gdpo_squeeze_*_${SLURM_JOB_ID}.out
# Fine-tuned checkpoints:  experiments/results/gdpo_connectivity/<archive>/gdpo_connected.ckpt
