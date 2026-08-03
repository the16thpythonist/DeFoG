#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=moses_kek_earlycheck_%j.out

# JUPITER: early read on the kekulized MOSES model after link 1 of 3.
# Submitted with --dependency=afterany:<link1>, so it runs alongside link 2 on
# a separate allocation rather than delaying the chain.
#
# WHAT THIS CAN AND CANNOT CONCLUDE
# Link 1 gives roughly 33 of the 100-epoch cosine horizon, so the kekulized
# model is undertrained AND its LR schedule has not annealed. The aromatic
# baseline it is compared against is FULLY trained (100 epochs, validity
# ~0.90 at n=2048, 0.875 at n=1024).
#
# That makes this an ASYMMETRIC test, and it is worth being explicit:
#
#   kekulized@33ep >= aromatic@100ep  ->  CONFIRMS the hypothesis with margin.
#                                         A third-trained model already beating
#                                         a fully trained one is not noise.
#   kekulized@33ep <  aromatic@100ep  ->  INCONCLUSIVE, not a refutation.
#                                         Wait for the full chain.
#
# There is no matched-epoch baseline available: the aromatic run's checkpoint
# carries no 'validity' history (its generation probe never recorded one --
# best_validity is -1.0), so a 33-epoch aromatic number does not exist and
# would cost a second training run to obtain.
#
# The 'kekulize' failure category should be exactly 0 for the new model. That is
# true BY CONSTRUCTION -- there is no AROMATIC bond class to fail on -- so it is
# a wiring check, not evidence. The evidence is what those 11.5 points became:
# genuine validity, or some other failure category.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

N=2048
STEPS=500
ETA=25          # the aromatic model's frozen deploy eta, for comparability.
                # NOT necessarily optimal for the kekulized model -- its own
                # sweep comes later. Reused here so the two rows differ in the
                # representation and nothing else.

KEK_CKPT="ckpts/moses_kek_seed42/best_model"
AROM_CKPT="ckpts/moses_e1_seed42/best_model"

if [ ! -f "${KEK_CKPT}.ckpt" ]; then
    echo "ERROR: ${KEK_CKPT}.ckpt missing -- did link 1 produce a checkpoint?"
    ls -la ckpts/moses_kek_seed42/ 2>/dev/null || echo "  (no such directory)"
    exit 1
fi

nvidia-smi --query-gpu=index,name --format=csv,noheader || true
python - <<'PY' || { echo "ERROR: CUDA preflight failed"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False"); sys.exit(1)
torch.zeros(8, device="cuda").sum().item()
print("CUDA preflight OK")
PY

echo "how far did link 1 get?"
python - <<'PY'
import torch
for tag, p in [("kekulized", "ckpts/moses_kek_seed42/last.ckpt"),
               ("aromatic ", "ckpts/moses_e1_seed42/last.ckpt")]:
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        print(f"  {tag}: epoch {ck.get('epoch')} / step {ck.get('global_step')}")
    except Exception as exc:
        print(f"  {tag}: {type(exc).__name__}: {exc}")
PY

echo
echo "================ KEKULIZED (link 1, ~1/3 trained) ================"
CUDA_VISIBLE_DEVICES=0 python -u scripts/diagnose_validity.py \
    --ckpt "${KEK_CKPT}" --dataset moses --representation kekulized_v2 \
    --n ${N} --steps ${STEPS} --eta ${ETA} \
    --dump-smiles "moses_kek_early_${SLURM_JOB_ID}.smi" \
    --out "moses_kek_early_${SLURM_JOB_ID}.json" 2>&1 | grep -vE "^\[[0-9]" &

echo "================ AROMATIC (fully trained, reference) ============="
CUDA_VISIBLE_DEVICES=1 python -u scripts/diagnose_validity.py \
    --ckpt "${AROM_CKPT}" --dataset moses \
    --n ${N} --steps ${STEPS} --eta ${ETA} \
    --dump-smiles "moses_arom_ref_${SLURM_JOB_ID}.smi" \
    --out "moses_arom_ref_${SLURM_JOB_ID}.json" 2>&1 | grep -vE "^\[[0-9]" &

wait

echo
echo "======================== SIDE BY SIDE ==========================="
python - "$SLURM_JOB_ID" <<'PY'
import json, sys
jid = sys.argv[1]
rows = [("kekulized@link1", f"moses_kek_early_{jid}.json"),
        ("aromatic@100ep",  f"moses_arom_ref_{jid}.json")]
data = {}
for name, path in rows:
    try:
        data[name] = json.load(open(path))
    except Exception as exc:
        print(f"{name}: could not read {path} ({exc})")
cats = ["ok", "kekulize", "valence", "disconnected", "other_sanitize",
        "decode_failed"]
print(f"{'category':16s}" + "".join(f"{n:>18s}" for n in data))
for c in cats:
    line = f"{c:16s}"
    for n, d in data.items():
        tot = max(1, d["n"])
        line += f"{d['counts'].get(c, 0):>10d} {d['counts'].get(c, 0)/tot:>7.4f}"
    print(line)
print()
if len(data) == 2:
    k, a = data["kekulized@link1"], data["aromatic@100ep"]
    kv = k["counts"].get("ok", 0) / max(1, k["n"])
    av = a["counts"].get("ok", 0) / max(1, a["n"])
    print(f"valid+connected: kekulized {kv:.4f} vs aromatic {av:.4f} "
          f"({kv - av:+.4f})")
    if kv >= av:
        print("  -> CONFIRMS: a third-trained model already matches or beats the "
              "fully trained aromatic baseline.")
    else:
        print("  -> INCONCLUSIVE: the kekulized model is undertrained "
              "(~33 of 100 epochs, LR not annealed). Not a refutation; wait for "
              "the full chain before drawing any conclusion.")
PY

echo
echo "NEXT: score the two .smi files against the validation reference with"
echo "scripts/e1_metrics.py in .venv_metrics for FCD. Note the kekulized model"
echo "is undertrained, so a worse FCD here is expected and not yet meaningful."
