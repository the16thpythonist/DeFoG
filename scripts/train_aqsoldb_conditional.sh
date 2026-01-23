#!/bin/bash
# AutoSlurmX script for training DeFoG on AqSolDB with logP conditioning
# Run this script from the DeFoG repository root directory
#
# Usage:
#   ./scripts/train_aqsoldb_conditional.sh
#
# Prerequisites:
#   1. Install AutoSlurm: pip install git+https://github.com/aimat-lab/AutoSlurm.git
#   2. Configure your conda environment in ~/.config/auto_slurm/general_config.yaml
#   3. Run the logP computation script first:
#      conda activate defog
#      python scripts/add_logp_to_csv.py
#   4. Delete old processed data (data will be regenerated with logP values):
#      rm -rf data/aqsoldb/aqsoldb_pyg/processed/
#
# Notes:
#   - Uses HAICORE with 1 GPU (A100)
#   - Training time: ~200 epochs
#   - Conditional generation: continuous logP (Crippen)
#   - logP verification is performed during sampling
#
# Available conditioning targets (set via general.target):
#   - 'logp'           : continuous logP conditioning (default)
#   - 'logp_binary'    : binary high/low lipophilicity
#   - 'solubility'     : continuous solubility
#   - 'solubility_binary' : binary high/low solubility

set -e

# Change to the src directory (required for DeFoG)
cd "$(dirname "$0")/../src"

# Launch training with AutoSlurmX on HAICORE
aslurmx -cn haicore_1gpu \
    -o conda_env=defog,job_name=aqsoldb_logp \
    cmd python main.py +experiment=aqsoldb_conditional dataset=aqsoldb

echo "Job submitted! Check status with: squeue -u \$USER"
