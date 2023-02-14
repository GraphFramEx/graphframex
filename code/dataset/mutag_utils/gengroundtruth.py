import networkx as nx
from rdkit import Chem
import numpy as np

MUTAG_SMILES = [
    "N(=O)[O]",
    # "[NH2]",
    "C1=CC=C(C=C1)N",
    "C1=CC=C(C=C1)[N+](=O)O",
    "N=O",
    "N=N",
]

MUTAG_NODE_LABELS = {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"}
MUTAG_NODE_LABELS_INV = {"C": 0, "N": 1, "O": 2, "F": 3, "I": 4, "Cl": 5, "Br": 6}


def smiles_to_nx(smiles):
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(
            atom.GetIdx(),
            atomic_num=atom.GetAtomicNum(),
            is_aromatic=atom.GetIsAromatic(),
            atom_symbol=atom.GetSymbol(),
            label=MUTAG_NODE_LABELS_INV[atom.GetSymbol()],
        )
    for bond in mol.GetBonds():
        G.add_edge(
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType()
        )
    return G


def get_ground_truth_mol(dataset_name):
    if dataset_name == "mutag":
        G_true_list = []
        for smiles in MUTAG_SMILES:
            G_true_list.append(smiles_to_nx(smiles))
        return G_true_list
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))


"""Detect the edges that connect N and O atoms."""
"""def get_true_mask(data):
    important_atoms = ["N", "O"]
    important_edges = [(1,2),(2,1)]
    for i, u, v in data.edge_index.T:
        if (u,v) in important_edges:
            data.edge_attr[u,v] = 1
    targets = [MUTAG_NODE_LABELS_INV[atom] for atom in important_atoms]
    nmb_targets = []
    for i, target in enumerate(targets):
        nmb_targets.append(np.where(data.x[:,target]==1)[0])
    for 
    nb_clss = 7
    targets = [MUTAG_NODE_LABELS_INV[atom] for atom in important_atoms]
    one_hot_feature = np.eye(nb_clss)[targets]
    print("one_hot_feature: ", one_hot_feature)
    np.argmax(data.x"""
