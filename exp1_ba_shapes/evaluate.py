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
from utils import list_to_dict

import torch_geometric.data
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, to_undirected

def topk_edges_directed(edge_mask, edge_index, num_top_edges):
    indices = (-edge_mask).argsort()
    top = np.array([], dtype='int')
    i = 0
    list_edges = np.sort(edge_index.T, axis=1)
    while len(top)<num_top_edges:
        subset = indices[num_top_edges*i:num_top_edges*(i+1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]], dtype=np.int32)
        i+=1
    return top[:num_top_edges]

#transform edge_mask:
# normalisation
# sparsity
# hard or soft
def transform_mask(mask, sparsity=0.7, normalize=True, hard_mask=False):
    if sparsity is not None:
        mask = control_sparsity(mask, sparsity)
    if normalize:
        mask = normalize(mask)
    if hard_mask:
        mask = np.where(mask>0, 1, 0)
    return(mask)

def mask_to_shape(mask, data, num_top_edges):
    indices = topk_edges_directed(mask, data.edge_index, num_top_edges)
    new_mask = np.zeros(len(mask))
    new_mask[indices] = 1
    return mask

def control_sparsity(mask, sparsity):
    r"""
        :param edge_mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
    """
    mask_len = len(mask)
    split_point = int((1 - sparsity) * mask_len)
    unimportant_indices = (-mask).argsort()[split_point:]
    mask[unimportant_indices] = 0
    return mask

def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

##### Accuracy #####
def get_explanation(data, edge_mask, num_top_edges):
    edge_mask = mask_to_shape(edge_mask, data, num_top_edges)
    indices = np.where(edge_mask>0)[0]
    explanation = data.edge_index[:, indices]
    edge_weights = edge_mask[indices]
    subx = np.unique(explanation)
    data_expl = Data(x=subx, edge_index=explanation)
    G_expl = to_networkx(data_expl)
    return G_expl.to_undirected()

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


def get_scores(G1, G2):
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

def get_accuracy(data, edge_mask, node_idx, num_top_edges, is_hard_mask=False):
    G_true, role, true_edge_mask = get_ground_truth_ba_shapes(node_idx, data)
    # nx.draw(G_true, cmap=plt.get_cmap('viridis'), node_color=role, with_labels=True, font_weight='bold')
    G_expl = get_explanation(data, edge_mask, num_top_edges, hard_explanation=True, sparsity=None)
    # plt.figure()
    # nx.draw(G_expl, with_labels=True, font_weight='bold')
    # plt.show()
    # plt.clf()
    recall, precision, f1_score, ged = get_scores(G_expl, G_true)
    fpr, tpr, thresholds = metrics.roc_curve(true_edge_mask, edge_mask, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    return {'recall': recall, 'precision': precision, 'f1_score': f1_score, 'ged': ged, 'auc': auc}


def eval_accuracy(data, edge_masks, list_node_idx, num_top_edges, is_hard_mask=False):
    n_test = len(list_node_idx)
    scores = []

    for i in range(n_test):
        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]
        entry = get_accuracy(data, edge_mask, node_idx, num_top_edges, is_hard_mask)
        scores.append(entry)

    scores = list_to_dict(scores)
    accuracy_scores = {k: np.mean(v) for k, v in scores.items()}
    return (accuracy_scores)




##### Fidelity #####
def eval_related_pred(model, data, edge_masks, list_node_idx, args):
    zero_mask = torch.zeros(data.edge_index.shape[1])

    ori_preds = model(data.x, data.edge_index)
    zero_mask_preds = model(x=data.x, edge_index=data.edge_index, edge_weight=zero_mask)

    n_test = len(list_node_idx)
    related_preds = []

    for i in range(n_test):

        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]

        edge_mask = transform_mask(edge_mask, args.sparsity, args.normalize, args.hard_mask)

        indices = np.where(edge_mask>0)[0]
        indices_inv = [i for i in range(len(edge_mask)) if i not in indices]

        masked_edge_index = data.edge_index[:, indices]
        maskout_edge_index = data.edge_index[:, indices_inv]


        ### reduce data.x to subx
        masked_x = np.unique(masked_edge_index)
        maskout_x = np.unique(maskout_edge_index)

        masked_preds = model(masked_x, masked_edge_index)
        maskout_preds = model(maskout_x, maskout_edge_index)

        # if hard mask is False, we can take edge_weights
        # masked_preds = model(x=masked_x, edge_index=data.edge_index, edge_weight=edge_mask)
        # maskout_preds = model(x=maskout_x, edge_index=data.edge_index, edge_weight=1 - edge_mask)


        ori_probs = softmax(ori_preds[node_idx].detach().numpy())
        masked_probs = softmax(masked_preds[node_idx].detach().numpy())
        maskout_probs = softmax(maskout_preds[node_idx].detach().numpy())
        zero_mask_probs = softmax(zero_mask_preds[node_idx].detach().numpy())

        true_label = data.y[node_idx].detach().numpy()
        pred_label = np.argmax(ori_probs)
        assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

        related_preds.append({'zero': zero_mask_probs,
                              'masked': masked_probs,
                              'maskout': maskout_probs,
                              'origin': ori_probs,
                              'true_label': true_label})

    related_preds = list_to_dict(related_preds)
    return related_preds


def fidelity_acc(related_preds):
    labels = related_preds['true_label']
    ori_labels = np.argmax(related_preds['origin'], axis=1)
    unimportant_labels = np.argmax(related_preds['maskout'], axis=1)
    p_1 = np.array(ori_labels == labels).astype(int)
    p_2 = np.array(unimportant_labels == labels).astype(int)
    drop_probability = p_1 - p_2
    return drop_probability.mean().item()


def fidelity_acc_inv(related_preds):
    labels = related_preds['true_label']
    ori_labels = np.argmax(related_preds['origin'], axis=1)
    important_labels = np.argmax(related_preds['masked'], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = p_1 - p_2
    return drop_probability.mean().item()


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_prob(related_preds):
    labels = related_preds['true_label']
    ori_probs = np.choose(labels, related_preds['origin'].T)
    unimportant_probs = np.choose(labels, related_preds['maskout'].T)
    drop_probability = ori_probs - unimportant_probs
    return drop_probability.mean().item()

# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_prob_inv(related_preds):
    labels = related_preds['true_label']
    ori_probs = np.choose(labels, related_preds['origin'].T)
    important_probs = np.choose(labels, related_preds['masked'].T)
    drop_probability = ori_probs - important_probs
    return drop_probability.mean().item()

def eval_fidelity(related_preds):
    fidelity_scores = {
        "fidelity_acc+": fidelity_acc(related_preds),
        "fidelity_acc-": fidelity_acc_inv(related_preds),
        "fidelity_prob+": fidelity_prob(related_preds),
        "fidelity_prob-": fidelity_prob_inv(related_preds),
    }
    return fidelity_scores