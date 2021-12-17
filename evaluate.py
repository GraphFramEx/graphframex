import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from dataset import house


def get_explanation(data, edge_mask, num_top_edges):
    # values,indices = edge_mask.topk(num_top_edges)
    indices = (-edge_mask).argsort()[:num_top_edges]
    explanation = data.edge_index[:, indices]

    G_expl = nx.Graph()
    G_expl.add_nodes_from(np.unique(explanation))

    for i, (u, v) in enumerate(explanation.t().tolist()):
        G_expl.add_edge(u, v)

    return (G_expl)

def get_ground_truth_ba_shapes(node):
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    start = ground_truth[0]
    graph, role = house(start, role_start=1)
    return graph, role


def scores(G1, G2):
    g_int = nx.intersection(G1, G2)
    g_int.remove_nodes_from(list(nx.isolates(g_int)))

    n_tp = g_int.number_of_edges()
    n_fp = len(G1.edges() - g_int.edges())
    n_fn = len(G2.edges() - g_int.edges())

    precision = n_tp / (n_tp + n_fp)
    recall = n_tp / (n_tp + n_fn)
    if n_tp == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    ged = nx.graph_edit_distance(G1, G2)

    return recall, precision, f1_score, ged


def evaluate(node_idx, data, edge_mask, num_top_edges):
    G_true, role = get_ground_truth_ba_shapes(node_idx)
    # nx.draw(G_true, cmap=plt.get_cmap('viridis'), node_color=role, with_labels=True, font_weight='bold')
    G_expl = get_explanation(data, edge_mask, num_top_edges)
    plt.figure()
    nx.draw(G_expl,  with_labels=True, font_weight='bold')
    plt.show()
    plt.clf()
    return scores(G_expl, G_true)

def evaluate_subgraph(node_idx, G_expl):
    G_true, role = get_ground_truth_ba_shapes(node_idx)
    nx.draw(G_true,  with_labels=True, font_weight='bold')
    plt.figure()
    nx.draw(G_expl,  with_labels=True, font_weight='bold')
    plt.show()
    plt.clf()
    return scores(G_expl, G_true)