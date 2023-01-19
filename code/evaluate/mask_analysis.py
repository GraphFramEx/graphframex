#%%
import matplotlib
import matplotlib.pyplot as plt
import sys

path = "/cluster/home/kamara/Explain/code"
sys.path.insert(0, path)
from evaluate.mask_utils import clean_all_masks, from_mask_to_nxsubgraph
from dataset.gen_real import load_data_real
from utils.path import MASK_DIR
import os
import pickle
import torch
import networkx as nx
import numpy as np
from utils.parser_utils import arg_parse


#%%
def get_mask_path(args):
    mask_path = MASK_DIR + args.dataset + "/" + args.explainer_name + "/"
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    return mask_path


def get_mask_properties(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data_real(args, device)

    mask_filename = (
        get_mask_path(args)
        + args.dataset
        + "_"
        + args.explainer_name
        + "_model_"
        + args.model
        + "_phenfocus_"
        + str(args.true_label_as_target)
        + "_test_"
        + str(args.num_test)
        + "_seed"
        + str(args.seed)
        + ".pkl"
    )
    with open(mask_filename, "rb") as f:
        w_list = pickle.load(f)
        node_indices, edge_masks, node_feat_masks, Time = tuple(w_list)
    edge_masks, node_feat_masks = clean_all_masks(edge_masks, node_feat_masks, args)
    for i, node_index in enumerate(node_indices):
        subgraph, relabeled_node_index = from_mask_to_nxsubgraph(
            edge_masks[i], node_index, data
        )
        connected_subgraph = subgraph.subgraph(
            nx.node_connected_component(subgraph.to_undirected(), relabeled_node_index)
        )
        edge_weights = nx.get_edge_attributes(subgraph, "weight")
        print("is subgraph connected:", nx.is_weakly_connected(subgraph))
        print(
            "is connected subgraph connected:",
            nx.is_weakly_connected(connected_subgraph),
        )

        # Eccentricity: the maximum distance from v=relabeled_node_index (the target node) to all other nodes in G
        ecc = nx.eccentricity(connected_subgraph, v=relabeled_node_index)
        # Diameter: the maximum eccentricity
        diam = nx.diameter(connected_subgraph)
        # Information centrality: Current-flow closeness centrality = variant of closeness centrality based on effective resistance between nodes in a network.
        centralities = nx.information_centrality(
            connected_subgraph.to_undirected(), weight="weights"
        )
        centralities_val = np.array([val for val in centralities.values()])
        ranks = np.argsort(centralities_val)[::-1]
        # Avg node connectivity: the average of local node connectivity over all pairs of nodes of G.
        avg_node_connectivity = nx.average_node_connectivity(subgraph)
        # Weighted clustering: the geometric average of the subgraph edge weights
        avg_clustering = nx.average_clustering(connected_subgraph, weight="weights")
        mask_property = {
            "dataset": args.dataset,
            "explainer": args.explainer_name,
            "node_id": node_index,
            "num_nodes": subgraph.number_of_nodes(),
            "num_edges": subgraph.number_of_edges(),
            "node_eccentricity": ecc,
            "diameter": diam,
            "node_centrality": centralities[0],
            "node_centrality_rank": np.where(ranks == relabeled_node_index)[0][0],
            "avg_node_connectivity": avg_node_connectivity,
            "avg_clustering": avg_clustering,
        }
        print(mask_property)


#%%

args = arg_parse()
args.dataset, args.E, args.NF = "cora", True, True
get_mask_properties(args)


# %%
