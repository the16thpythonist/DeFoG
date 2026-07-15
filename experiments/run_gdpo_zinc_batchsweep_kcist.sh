#!/bin/bash
#SBATCH --job-name=gdpo_zinc_batchsweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_zinc_batchsweep_%j.out

# GDPO connectivity fine-tune -- 4-way ROLLOUT BATCH-SIZE sweep on one node
# (4x RTX 4090, 24 GB), one batch size K per GPU. Trains the disconnected-fragment
# failure mode out of the unconditional ZINC model (reward: connected&valid=1, else 0),
# CONTINUING from the round-1 fine-tuned checkpoint. Fixed 120 iterations per arm;
# only the rollout batch size K differs (100/200/300/400). Question: does a bigger
# rollout batch push disconnection below the ~7% floor, or is the floor KL-set?
#
# PREREQUISITE -- the round-1 checkpoint is a gitignored local artifact, so it is NOT
# in the repo. Copy it to KCIST first (38 MB), e.g. from the machine that produced it:
#   scp experiments/_gdpo_zinc_connectivity/gdpo_connected.ckpt kcist:~/gdpo_zinc_round1.ckpt
# then (on KCIST):  git pull  &&  sbatch experiments/run_gdpo_zinc_batchsweep_kcist.sh
# Override the path with:  CKPT=/path/to/ckpt sbatch experiments/run_gdpo_zinc_batchsweep_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/gdpo_zinc_round1.ckpt}"   # round-1 fine-tuned ZINC checkpoint
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint '$CKPT' not found. Copy the round-1 gdpo_connected.ckpt to KCIST"
  echo "       (see header), or pass CKPT=/path/to/ckpt."
  exit 1
fi

KS=(100 200 300 400)
echo "GDPO ZINC batch-size sweep K=${KS[*]} | 120 iters | continue from $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  K=${KS[$i]}
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity__aqsoldb.py \
      --ckpt "$CKPT" \
      --rollout-size "$K" \
      --iterations 120 \
      --sample-steps 100 --subsample-steps 12 --minibatch-size 32 \
      --eta 0 --reduction sum --lr 2e-5 --kl-coef 0.3 --ema-decay 0.9 \
      --eval-samples 2048 --eval-steps 100 --ckpt-every 20 --seed 0 \
      --outdir "experiments/_gdpo_zinc_batchsweep/K${K}" \
      > "gdpo_zinc_K${K}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm K=$K on GPU $i (pid $!)"
  sleep 5
done

wait
echo "all 4 arms finished at $(date)"
# Compare the AFTER lines:  grep -H 'AFTER' gdpo_zinc_K*_${SLURM_JOB_ID}.out
# Per-arm trace + snapshots:  experiments/_gdpo_zinc_batchsweep/K<K>/{history.json,ckpts/}
