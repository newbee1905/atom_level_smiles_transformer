# benchmark_smiles.py
import pytest
import numpy as np
from rdkit import Chem

SMILES_SAMPLE = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin


def rdkit_default(mol):
	return Chem.MolToSmiles(mol, doRandom=True, canonical=False)


def restricted_renumber(mol):
	atom_order = list(range(mol.GetNumAtoms()))
	np.random.shuffle(atom_order)
	new_mol = Chem.RenumberAtoms(mol, atom_order)
	return Chem.MolToSmiles(new_mol)


@pytest.mark.parametrize("smiles", [SMILES_SAMPLE])
def test_rdkit_default(benchmark, smiles):
	mol = Chem.MolFromSmiles(smiles)
	benchmark(rdkit_default, mol)


@pytest.mark.parametrize("smiles", [SMILES_SAMPLE])
def test_restricted_renumber(benchmark, smiles):
	mol = Chem.MolFromSmiles(smiles)
	benchmark(restricted_renumber, mol)
