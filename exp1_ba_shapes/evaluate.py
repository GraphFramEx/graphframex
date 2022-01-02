import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from dataset import house
from explainer import node_attr_to_edge
from sklearn import metrics
from scipy.special import softmax
import torch



def get_explanation(data, edge_mask, num_top_edges=6, is_hard_mask=False):
    if is_hard_mask:
        explanation = data.edge_index[:, np.where(edge_mask == 1)[0]]
    else:
        indices = (-edge_mask).argsort()[:num_top_edges]
        explanation = data.edge_index[:, indices]

    G_expl = nx.Graph()
    G_expl.add_nodes_from(np.unique(explanation))

    for i, (u, v) in enumerate(explanation.t().tolist()):
        G_expl.add_edge(u, v)

    return (G_expl)

def get_ground_truth_ba_shapes(node, data):
    base = [0, 1, 2, 3, 4]
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    start = ground_truth[0]
    graph, role = house(start, role_start=1)

    true_node_mask = np.zeros(data.edge_index.shape[1])
    true_node_mask[ground_truth] = 1
    true_edge_mask = node_attr_to_edge(data.edge_index, true_node_mask)

    return graph, role, true_edge_mask


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


def evaluate(node_idx, data, edge_mask, num_top_edges, is_hard_mask=False):
    G_true, role, true_edge_mask = get_ground_truth_ba_shapes(node_idx, data)
    # nx.draw(G_true, cmap=plt.get_cmap('viridis'), node_color=role, with_labels=True, font_weight='bold')
    G_expl = get_explanation(data, edge_mask, num_top_edges, is_hard_mask=is_hard_mask)
    plt.figure()
    nx.draw(G_expl,  with_labels=True, font_weight='bold')
    plt.show()
    plt.clf()
    recall, precision, f1_score, ged = scores(G_expl, G_true)
    fpr, tpr, thresholds = metrics.roc_curve(true_edge_mask, edge_mask, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    return recall, precision, f1_score, ged, auc


def add_related_pred(model, data, related_preds, edge_mask, node_idx, true_label=True):
    edge_mask = torch.Tensor(edge_mask)
    zero_mask = torch.zeros(edge_mask.size())

    ori_pred = model(data.x, data.edge_index)
    masked_pred = model(x=data.x, edge_index=data.edge_index, edge_weight=edge_mask)
    maskout_pred = model(x=data.x, edge_index=data.edge_index, edge_weight=1 - edge_mask)
    zero_mask_pred = model(x=data.x, edge_index=data.edge_index, edge_weight=zero_mask)

    ori_prob = softmax(ori_pred[node_idx].detach().numpy())
    masked_prob = softmax(masked_pred[node_idx].detach().numpy())
    maskout_prob = softmax(maskout_pred[node_idx].detach().numpy())
    zero_mask_prob = softmax(zero_mask_pred[node_idx].detach().numpy())

    if true_label:
        label = data.y[node_idx].detach().numpy()
    else:
        label = np.argmax(ori_prob)

    related_preds.append({'zero': zero_mask_prob[label],
                          'masked': masked_prob[label],
                          'maskout': maskout_prob[label],
                          'origin': ori_prob[label]})
    return related_preds


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations
def fidelity(related_preds):
    ori_probs = related_preds['origin']
    unimportant_probs = related_preds['maskout']
    drop_probability = ori_probs - unimportant_probs
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations
def fidelity_inv(related_preds):
    ori_probs = related_preds['origin']
    important_probs = related_preds['masked']
    drop_probability = ori_probs - important_probs
    return drop_probability.mean().item()

