""" fidelity.py
    Compute Fidelity+ and Fidelity- scores.
"""

import numpy as np
import torch
from exp_mutag.graph_utils import compute_masked_edges

from .code.gnn.eval import get_proba, get_true_labels_gc, gnn_preds_gc
from .code.utils.gen_utils import list_to_dict


def eval_related_pred_nc(model, data, edge_masks, list_node_idx, device):
    related_preds = []
    data = data.to(device)
    ori_ypred = model(data.x, data.edge_index)
    ori_yprob = get_proba(ori_ypred)

    n_test = len(list_node_idx)

    for i in range(n_test):
        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]
        mask_sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)

        indices = np.where(edge_mask > 0)[0]
        indices_inv = [i for i in range(len(edge_mask)) if i not in indices]

        masked_edge_index = data.edge_index[:, indices].to(device)
        maskout_edge_index = data.edge_index[:, indices_inv].to(device)

        masked_ypred = model(data.x, masked_edge_index)
        masked_yprob = get_proba(masked_ypred)

        maskout_ypred = model(data.x, maskout_edge_index)
        maskout_yprob = get_proba(maskout_ypred)

        ori_probs = ori_yprob[node_idx].detach().cpu().numpy()
        masked_probs = masked_yprob[node_idx].detach().cpu().numpy()
        maskout_probs = maskout_yprob[node_idx].detach().cpu().numpy()

        true_label = data.y[node_idx].cpu().numpy()
        pred_label = np.argmax(ori_probs)
        # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

        related_preds.append({'node_idx': node_idx,
                              'masked': masked_probs,
                              'maskout': maskout_probs,
                              'origin': ori_probs,
                              'mask_sparsity': mask_sparsity,
                              'expl_edges': (edge_mask != 0).sum(),
                              'true_label': true_label,
                              'pred_label': pred_label})

    related_preds = list_to_dict(related_preds)
    return related_preds


def eval_related_pred_gc(model, dataset, edge_index_set, edge_masks_set, device, args):
    n_graphs = len(np.hstack(edge_masks_set))
    n_bs = len(edge_masks_set)

    mask_sparsity = 0
    expl_edges = 0
    for edge_mask in np.hstack(edge_masks_set):
        mask_sparsity += 1.0 - (edge_mask != 0).sum() / len(edge_mask)
        expl_edges += (edge_mask != 0).sum()
    mask_sparsity /= n_graphs
    expl_edges /= n_graphs

    #related_preds = []

    ori_ypred = gnn_preds_gc(model, dataset, edge_index_set, args)
    ori_yprob = get_proba(ori_ypred)

    masked_edge_index_set, maskout_edge_index_set = compute_masked_edges(
        edge_masks_set, edge_index_set, device)

    masked_ypred = gnn_preds_gc(model, dataset, masked_edge_index_set, args)
    masked_yprob = get_proba(masked_ypred)

    maskout_ypred = gnn_preds_gc(model, dataset, maskout_edge_index_set, args)
    maskout_yprob = get_proba(maskout_ypred)

    #true_label = [data["label"].long().numpy() for batch_idx, data in enumerate(dataset)]
    #pred_label = get_labels(ori_ypred)
    # print(true_label)
    # print(pred_label)
    # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

    related_preds = {'masked': masked_yprob,
                     'maskout': maskout_yprob,
                     'origin': ori_yprob,
                     'mask_sparsity': mask_sparsity,
                     'expl_edges': expl_edges,
                     'true_label': get_true_labels_gc(dataset)
                     }
    #related_preds = list_to_dict(related_preds)
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
