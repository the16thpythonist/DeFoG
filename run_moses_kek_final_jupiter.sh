#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --output=moses_kek_final_%j.out

# JUPITER: evaluate the FINISHED kekulized MOSES model -- all four seeds at the
# full 100-epoch cosine horizon, one seed per GPU.
#
# The early check (job 1224996) read seed 42 at ~36 epochs and already beat the
# fully trained aromatic model on every axis. This is the completed measurement,
# with the seed spread that a single checkpoint cannot give.
#
# CONFIG CAVEAT, stated because it limits what these numbers mean:
# eta=25 is the AROMATIC model's frozen value, reused so the comparison isolates
# the representation. It is almost certainly not this model's optimum -- the
# ZINC RL model's own sweep moved its best config, and there is no reason to
# expect kekulized MOSES to share the aromatic model's. These numbers are
# therefore a LOWER BOUND on the representation change, and NOT an E1 table row.
# An E1 row needs this model's own validation sweep plus one frozen test pass.
#
# The aromatic reference is re-measured in the same job rather than quoted, so
# both sides share sampler, n and RDKit version.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

N=2048
STEPS=500
ETA=25
SEEDS=(42 43 44 45)

for s in "${SEEDS[@]}"; do
    if [ ! -f "ckpts/moses_kek_seed${s}/best_model.ckpt" ]; then
        echo "ERROR: ckpts/moses_kek_seed${s}/best_model.ckpt missing"; exit 1
    fi
done

nvidia-smi --query-gpu=index,name --format=csv,noheader || true
python - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK")
PY

echo "epochs reached per seed:"
python - <<'PY'
import torch
for s in (42, 43, 44, 45):
    try:
        ck = torch.load(f"ckpts/moses_kek_seed{s}/last.ckpt", map_location="cpu",
                        weights_only=False)
        print(f"  seed {s}: epoch {ck.get('epoch')}")
    except Exception as exc:
        print(f"  seed {s}: {type(exc).__name__}")
PY

for i in 0 1 2 3; do
    s=${SEEDS[$i]}
    CUDA_VISIBLE_DEVICES=$i python -u scripts/diagnose_validity.py \
        --ckpt "ckpts/moses_kek_seed${s}/best_model" --dataset moses \
        --representation kekulized_v2 \
        --n ${N} --steps ${STEPS} --eta ${ETA} \
        --dump-smiles "moses_kek_final_seed${s}_${SLURM_JOB_ID}.smi" \
        --out "moses_kek_final_seed${s}_${SLURM_JOB_ID}.json" \
        > "moses_kek_final_seed${s}_${SLURM_JOB_ID}.out" 2>&1 &
    echo "launched seed ${s} on GPU ${i}"
    sleep 3
done
wait

echo
echo "=== kekulized, all seeds, n=${N} ==="
python - "$SLURM_JOB_ID" <<'PY'
import json, sys
jid = sys.argv[1]
rows = []
for s in (42, 43, 44, 45):
    try:
        d = json.load(open(f"moses_kek_final_seed{s}_{jid}.json"))
    except Exception as exc:
        print(f"  seed {s}: unreadable ({exc})"); continue
    n = max(1, d["n"]); c = d["counts"]
    rows.append((s, c.get("ok", 0) / n, c.get("kekulize", 0) / n,
                 c.get("valence", 0) / n, c.get("disconnected", 0) / n))
print(f"{'seed':>5s}{'valid':>10s}{'kekulize':>10s}{'valence':>9s}{'disconn':>9s}")
for s, ok, kek, val, dis in rows:
    print(f"{s:>5d}{ok:>10.4f}{kek:>10.4f}{val:>9.4f}{dis:>9.4f}")
if rows:
    import statistics as st
    v = [r[1] for r in rows]
    print(f"\nvalidity mean {st.mean(v):.4f}"
          + (f" +- {st.stdev(v):.4f}" if len(v) > 1 else ""))
    print("aromatic base, same sampler and n, was 0.8843 (job 1224996)")
PY

echo
echo "NEXT: score the .smi files against the validation reference for FCD."
echo "Then, for an E1 row: this model's OWN sampling sweep, then one frozen"
echo "test pass. eta=25 here is inherited from the aromatic model and is a"
echo "lower bound, not a tuned configuration."
