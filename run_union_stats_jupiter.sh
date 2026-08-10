#!/bin/bash
#SBATCH --account=aimatchem
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=union_stats_%j.out

# JUPITER: compute the KEKULIZED graph statistics for the ZINC∪ChEMBL union.
#
# This is CPU-only work that happens to run here because the node has 288 Grace
# cores and the data is already staged -- ~100M molecules take ~110 min on a
# 16-core desktop and a few minutes here. Short walltime, since the partition is
# whole-node exclusive.
#
# WHY IT IS REQUIRED: prepare_smiles_union.py writes an AROMATIC union_stats.json
# (4 bond types incl. AROMATIC). With noise_type="marginal" the edge marginals ARE
# the prior the model denoises from, and dropping the AROMATIC class moves ~3.5%
# of all atom pairs into single/double. Training kekulized from the aromatic prior
# would be denoising from the wrong distribution. train_chembl_ddp.py refuses to
# start on the mismatch rather than doing it silently.
#
# Counts run through smiles_to_pyg_data -- the same encoder training uses -- so the
# marginals cannot drift from what the model sees, and any molecule the encoder
# rejects is excluded from the prior exactly as it is excluded from training.

cd "$SLURM_SUBMIT_DIR"
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1
source .venv_jupiter/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1

DATA_DIR="data/zinc_chembl_union"
PREFIX="union"

echo "union kekulized stats @ $(date)"
python -u scripts/compute_graph_stats.py \
    --dataset chembl --representation kekulized_v2 \
    --smiles "$DATA_DIR/${PREFIX}_train.smiles" \
    --out "$DATA_DIR/${PREFIX}_kek_stats.json" \
    --workers 200
echo "union kekulized stats done @ $(date)"
