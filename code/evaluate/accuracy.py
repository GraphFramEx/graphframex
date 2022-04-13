""" accuracy.py
    Compute Accuracy scores when groundtruth is avaiable (synthetic datasets)
"""

import networkx as nx
import numpy as np
import torch
from utils.plot_utils import plot_expl_nc
from dataset.syn_utils.gengroundtruth import get_ground_truth
from utils.gen_utils import list_to_dict
from evaluate.mask_utils import mask_to_shape


def get_explanation(data, edge_mask, args, top_acc):
    if top_acc:
        # indices = (-edge_mask).argsort()[:kwargs['num_top_edges']]
        edge_mask = mask_to_shape(edge_mask, data.edge_index, args.num_top_edges)
        indices = np.where(edge_mask > 0)[0]
    else:
        edge_mask = edge_mask.cpu().detach().numpy()
        indices = np.where(edge_mask > 0)[0]

    explanation = data.edge_index[:, indices]
    weights = edge_mask[indices]
    G_expl = nx.Graph()
    G_expl.add_nodes_from(np.unique(explanation.cpu()))
    for i, (u, v) in enumerate(explanation.t().tolist()):
        G_expl.add_edge(u, v)
    k = 0
    for u, v, d in G_expl.edges(data=True):
        d["weight"] = weights[k]
        k += 1
    G_masked = G_expl.copy()
    for u, v, d in G_masked.edges(data=True):
        d["weight"] = (G_expl[u][v]["weight"] + G_expl[v][u]["weight"]) / 2

    labels = data.y[np.unique(explanation.cpu())]
    k = 0
    for n, d in G_masked.nodes(data=True):
        d["label"] = labels[k]
        k += 1
    return G_masked


def get_scores(G1, G2):
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


def get_accuracy(data, edge_mask, node_idx, args, top_acc):
    G_true, role, true_edge_mask = get_ground_truth(node_idx, data, args)
    G_expl = get_explanation(data, edge_mask, args, top_acc)
    if eval(args.draw_graph):
        plot_expl_nc(G_expl, G_true, role, node_idx, args, top_acc)
    recall, precision, f1_score = get_scores(G_expl, G_true)
    # fpr, tpr, thresholds = metrics.roc_curve(true_edge_mask, edge_mask, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    return {"recall": recall, "precision": precision, "f1_score": f1_score}


def eval_accuracy(data, edge_masks, list_node_idx, args, top_acc=False):
    scores = []

    for i in range(args.num_test_final):
        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]
        entry = get_accuracy(data, edge_mask, node_idx, args, top_acc)
        scores.append(entry)

    scores = list_to_dict(scores)
    accuracy_scores = {k: np.mean(v) for k, v in scores.items()}
    return accuracy_scores
