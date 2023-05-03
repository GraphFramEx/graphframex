""" accuracy.py
    Compute Accuracy scores when groundtruth is avaiable (synthetic datasets)
"""

import networkx as nx
import numpy as np
from evaluate.mask_utils import mask_to_shape


def get_explanation_syn(data, edge_mask, num_top_edges, top_acc):
    """Create an explanation graph from the edge_mask.
    Args:
        data (Pytorch data object): the initial graph as Data object
        edge_mask (Tensor): the explanation mask
        top_acc (bool): if True, use the top_acc as the threshold for the edge_mask
    Returns:
        G_masked (networkx graph): explanatory subgraph
    """
    if top_acc:
        # indices = (-edge_mask).argsort()[:kwargs['num_top_edges']]
        edge_mask = mask_to_shape(edge_mask, data.edge_index, num_top_edges)
        indices = np.where(edge_mask > 0)[0]
    else:
        edge_mask = edge_mask.cpu().detach().numpy()
        indices = np.where(edge_mask > 0)[0]

    explanation = data.edge_index[:, indices]
    weights = edge_mask[indices]
    explanation = explanation.cpu().numpy()
    G_expl = nx.Graph()
    G_expl.add_nodes_from(np.unique(explanation))
    for i, (u, v) in enumerate(explanation.T.tolist()):
        G_expl.add_edge(u, v)
    k = 0
    for u, v, d in G_expl.edges(data=True):
        d["weight"] = weights[k]
        k += 1
    G_masked = G_expl.copy()
    for u, v, d in G_masked.edges(data=True):
        d["weight"] = (G_expl[u][v]["weight"] + G_expl[v][u]["weight"]) / 2

    LABELS = data.y.detach().cpu().numpy()
    labels = LABELS[np.unique(explanation)]
    k = 0
    for n, d in G_masked.nodes(data=True):
        d["label"] = labels[k]
        k += 1
    return G_masked


def get_scores(G1, G2):
    """Compute recall, precision, and f1 score of a graph.

    Args:
        G1 (networkx graph): ground truth graph
        G2 (networkx graph): explanation graph
    """
    G1, G2 = G1.to_undirected(), G2.to_undirected()
    g_int = nx.intersection(G1, G2)
    g_int.remove_nodes_from(list(nx.isolates(g_int)))

    n_tp = g_int.number_of_edges()
    n_fp = len(G1.edges() - g_int.edges())
    n_fn = len(G2.edges() - g_int.edges())

    if n_tp == 0:
        precision, recall = 0, 0
        f1_score = 0
    else:
        precision = n_tp / (n_tp + n_fp)
        recall = n_tp / (n_tp + n_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1_score


def get_best_scores(G1, G2_list):
    R, P, F1 = [], [], []
    for G2 in G2_list:
        recall, precision, f1_score = get_scores(G1, G2)
        R.append(recall)
        P.append(precision)
        F1.append(f1_score)
    i_best = np.argmax(F1)
    return R[i_best], P[i_best], F1[i_best]

