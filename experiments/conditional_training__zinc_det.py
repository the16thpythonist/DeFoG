"""
Conditional generation on ZINC 250k with logP and molecule size conditioning.

Extends the base conditional generation experiment to train on the ZINC 250k
dataset using two regression properties as conditions:
  - logP (octanol-water partition coefficient)
  - num_atoms (heavy atom count / molecule size)

Usage:
    python experiments/conditional_training__zinc_det.py
    python experiments/conditional_training__zinc_det.py --__TESTING__ True
    python experiments/conditional_training__zinc_det.py --EPOCHS 200 --GUIDANCE_SCALE 3.0
"""
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

_PROJECT_DIR = Path(__file__).parent.parent.resolve()

# ============================================================================
# Inherit from the base conditional generation experiment
# ============================================================================

experiment = Experiment.extend(
    "conditional_generation.py",
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)

# ============================================================================
# Parameter overrides
# ============================================================================

# :param CSV_PATH:
#     Path to the ZINC 250k dataset CSV file.
CSV_PATH: str = str(_PROJECT_DIR / "data" / "zinc_250k_rdkit.csv")

# :param PROPERTIES:
#     Two conditioning properties: logP and molecule size (num_atoms).
#     Both are regression targets with z-score normalization.
PROPERTIES: dict = {
    "logp": {
        "type": "regression",
        "callback": lambda mol: Crippen.MolLogP(mol),
        "target": 2.5,
    },
    "num_atoms": {
        "type": "regression",
        "callback": lambda mol: mol.GetNumHeavyAtoms(),
        "target": 20,
    },
}

# ============================================================================
# Testing overrides
# ============================================================================

@experiment.testing
def testing(e: Experiment):
    """Reduce parameters for quick testing on a small subset."""
    e.CSV_PATH = str(_PROJECT_DIR / "data" / "test_molecules.csv")
    e.EPOCHS = 2
    e.BATCH_SIZE = 4
    e.NUM_EVAL_SAMPLES = 5
    e.SAMPLE_STEPS = 3
    e.SAMPLE_VIS_EVERY_K = 1
    e.N_LAYERS = 2
    e.HIDDEN_DIM = 32
    e.HIDDEN_MLP_DIM = 64
    e.N_HEADS = 2
    # Override properties to match test CSV columns
    e.PROPERTIES = {
        "logP": {
            "type": "regression",
            "callback": lambda mol: Crippen.MolLogP(mol),
            "target": 2.5,
        },
    }


experiment.run_if_main()
