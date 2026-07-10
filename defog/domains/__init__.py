"""
Concrete :class:`~defog.core.domain.GraphDomain` implementations for specific
kinds of graphs. These may pull in domain-specific dependencies (e.g. RDKit for
molecules) and are kept out of ``defog.core`` so the core stays lightweight.
"""
from .molecule import MoleculeDomain, pyg_data_to_mol, mol_to_smiles

__all__ = ["MoleculeDomain", "pyg_data_to_mol", "mol_to_smiles"]
