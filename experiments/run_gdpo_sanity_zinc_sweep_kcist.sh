#!/bin/bash
#SBATCH --job-name=gdpo_sanity_sweep
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=gdpo_sanity_sweep_%j.out

# GENTLER-CONFIG SWEEP for the structural-sanity fine-tune on ZINC. The connectivity-
# tuned config (kl=0.2, 100 iters, lr=2e-5) OVER-OPTIMIZES this harder reward: it fixes
# weird rings (artifact 10%->1%) but collapses validity (95%->49%), so best-snapshot
# fell back to the input (no gain). These 4 arms hold validity while trimming rings:
#   A  kl=0.4              (higher KL trust region)
#   B  kl=0.8              (strong KL)
#   C  positive_only=True  (RAFT: only pull UP sane endpoints, never push valids into
#                           invalid space -- directly targets the collapse mechanism)
#   D  lr=1e-5, 60 iters   (slower, longer)
# All: moderate envelope (rings {3-8}), fine snapshots (every 10) + input-inclusive
# best-snapshot with the same-step validity-floor gate, eval@500 + FCD. One model per
# GPU on a small-partition node (4x RTX 4090).
#
# PREREQUISITES on KCIST: git pull; fcd_torch installed; ~/zinc_round1.ckpt +
#   data/zinc_250k_rdkit.csv present.
#   git pull && sbatch experiments/run_gdpo_sanity_zinc_sweep_kcist.sh

cd ~/Programming/DeFoG
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

mkdir -p experiments/results/gdpo_sanity   # pycomex prepare_path race guard

CK="$HOME/zinc_round1.ckpt"
REF="data/zinc_250k_rdkit.csv"
echo "GDPO SANITY SWEEP (ZINC, gentler configs, eval@500 + FCD) | $(date)"
ls -la "$CK" || { echo "MISSING $CK"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Pre-build the reference/envelope cache ONCE (stable hashlib key) so the 4 parallel
# arms all read it instead of each rebuilding + racing on the write.
echo "pre-building envelope cache..."
python -c "from experiments.gdpo_sanity import prepare_reference; prepare_reference('$REF','smiles',1.0,50,None,10000,12,seed=0)" \
    && echo "cache ready" || { echo "cache pre-build FAILED"; exit 1; }

LABELS=(A_kl0.4 B_kl0.8 C_raft D_lowlr)
KLS=(0.4 0.8 0.3 0.3)
ITERS=(40 40 40 60)
POS=(False False True False)
LRS=(2e-5 2e-5 2e-5 1e-5)

for i in 0 1 2 3; do
  NM=${LABELS[$i]}; KL=${KLS[$i]}; IT=${ITERS[$i]}; PO=${POS[$i]}; LR=${LRS[$i]}
  echo "launching $NM (kl=$KL iters=$IT positive_only=$PO lr=$LR) on GPU $i"
  # pycomex evals --PARAM values -> quote STRING literals; numbers/bools pass through.
  CUDA_VISIBLE_DEVICES=$i python -u experiments/gdpo_sanity.py \
      --__DEBUG__ False \
      --CKPT_PATH "'$CK'" \
      --REFERENCE_SMILES "'$REF'" \
      --ENVELOPE_QUANTILE 1.0 --RING_MIN_COUNT 50 --REQUIRE_CONNECTED True \
      --KL_COEF $KL --KL_ANCHOR "'fixed'" --ADVANTAGE_MODE "'grpo'" --REDUCTION "'sum'" \
      --POSITIVE_ONLY $PO --LAMBDA_EDGE 1 --ROUNDS 1 --ITERATIONS $IT \
      --SAMPLE_STEPS 100 --SUBSAMPLE_STEPS 12 --TIME_DISTORTION "'polydec'" \
      --ROLLOUT_SIZE 128 --MINIBATCH_SIZE 16 --ETA 0.0 --LR $LR --EMA_DECAY 0.9 \
      --CKPT_EVERY 10 --SELECT_BEST True --SELECT_INCLUDE_INPUT True \
      --SELECT_EVAL_STEPS 100 --SELECT_EVAL_SAMPLES 1024 \
      --EVAL_SAMPLES 2048 --EVAL_STEPS 500 --COMPUTE_FCD True --FCD_REF_SAMPLES 10000 \
      --SEED 1 \
      > "gdpo_sanity_sweep_${NM}_${SLURM_JOB_ID}.out" 2>&1 &
  sleep 8
done

wait
echo "all arms finished at $(date)"
# Results:  grep -H 'SUMMARY' gdpo_sanity_sweep_*_${SLURM_JOB_ID}.out
