#!/usr/bin/env python3
"""Filter aqsoldb.csv to remove molecules with more than 30 atoms."""

import csv
from rdkit import Chem

# Read the CSV file
input_file = "/p/home/jusers/teufel1/juwels/Programming/DeFoG/aqsoldb.csv"
output_file = "/p/home/jusers/teufel1/juwels/Programming/DeFoG/aqsoldb_filtered.csv"

filtered_rows = []
removed_count = 0
total_count = 0

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames

    for row in reader:
        total_count += 1
        smiles = row['SMILES']
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            # Keep rows with invalid SMILES (shouldn't happen but just in case)
            filtered_rows.append(row)
            continue

        num_atoms = mol.GetNumAtoms()

        if num_atoms <= 30:
            filtered_rows.append(row)
        else:
            removed_count += 1

# Write filtered data
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(filtered_rows)

print(f"Original dataset size: {total_count} molecules")
print(f"Filtered dataset size: {len(filtered_rows)} molecules")
print(f"Removed: {removed_count} molecules (with > 30 atoms)")
print(f"Saved to: {output_file}")
