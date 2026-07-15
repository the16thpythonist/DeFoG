#!/bin/bash
#SBATCH --job-name=gdpo_zinc_bspre
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=gdpo_zinc_bspre_%j.out

# GDPO connectivity -- rollout BATCH-SIZE sweep from the PRETRAINED ZINC model
# (4x RTX 4090, one K per GPU). Companion to run_gdpo_zinc_batchsweep_kcist.sh,
# which continues from the already-fine-tuned round-1 checkpoint and overtrains
# (validity collapses to atom-soup past ~iter 100 because there is no connectivity
# headroom left). Starting from PRETRAINED restores headroom; 80 iters (not 120)
# reaches the floor before the late entropy blowup. This is the clean test of
# whether a bigger rollout batch reaches a LOWER disconnection floor.
#
# PREREQUISITE -- upload the pretrained ZINC checkpoint (not in the repo):
#   scp ~/Downloads/zinc_uncond_4e-4_best_model.ckpt kcist:~/zinc_uncond_pretrained.ckpt
# then (on KCIST):  git pull  &&  sbatch experiments/run_gdpo_zinc_batchsweep_pretrained_kcist.sh
# Override with:  CKPT=/path/to/ckpt sbatch ...

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CKPT="${CKPT:-$HOME/zinc_uncond_pretrained.ckpt}"   # PRETRAINED ZINC uncond checkpoint
if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint '$CKPT' not found. Upload zinc_uncond_4e-4_best_model.ckpt"
  echo "       to KCIST (see header), or pass CKPT=/path/to/ckpt."
  exit 1
fi

KS=(100 200 300 400)
echo "GDPO ZINC batch-size sweep (FROM PRETRAINED) K=${KS[*]} | 80 iters | base $CKPT | $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

for i in 0 1 2 3; do
  K=${KS[$i]}
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_connectivity__aqsoldb.py \
      --ckpt "$CKPT" \
      --rollout-size "$K" \
      --iterations 80 \
      --sample-steps 100 --subsample-steps 12 --minibatch-size 32 \
      --eta 0 --reduction sum --lr 2e-5 --kl-coef 0.3 --ema-decay 0.9 \
      --eval-samples 2048 --eval-steps 100 --ckpt-every 20 --seed 0 \
      --outdir "experiments/_gdpo_zinc_batchsweep_pre/K${K}" \
      > "gdpo_zinc_pre_K${K}_${SLURM_JOB_ID}.out" 2>&1 &
  echo "launched arm K=$K on GPU $i (pid $!)"
  sleep 5
done

wait
echo "all 4 arms finished at $(date)"
# Compare:  grep -H 'AFTER' gdpo_zinc_pre_K*_${SLURM_JOB_ID}.out
# Traces + snapshots:  experiments/_gdpo_zinc_batchsweep_pre/K<K>/{history.json,ckpts/}
