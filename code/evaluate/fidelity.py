""" fidelity.py
    Compute Fidelity+ and Fidelity- scores.
"""

import numpy as np
from sympy import re
import torch
from gnn.eval import gnn_preds_gc, gnn_preds_gc_batch
from utils.gen_utils import get_true_labels_gc_batch, list_to_dict, get_true_labels_gc, get_labels, get_proba
from utils.graph_utils import compute_masked_edges, compute_masked_edges_batch
from evaluate.mask_utils import get_size, get_sparsity


def eval_related_pred_nc(model, data, edge_masks, list_node_idx, device):
    related_preds = []
    data = data.to(device)
    ori_ypred = model(data.x, data.edge_index).cpu().detach().numpy()
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

        masked_ypred = model(data.x, masked_edge_index).cpu().detach().numpy()
        masked_yprob = get_proba(masked_ypred)

        maskout_ypred = model(data.x, maskout_edge_index).cpu().detach().numpy()
        maskout_yprob = get_proba(maskout_ypred)

        ori_probs = ori_yprob[node_idx]
        masked_probs = masked_yprob[node_idx]
        maskout_probs = maskout_yprob[node_idx]
        true_label = data.y[node_idx].cpu().numpy()
        pred_label = np.argmax(ori_probs)
        # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

        related_preds.append(
            {
                "node_idx": node_idx,
                "masked": masked_probs,
                "maskout": maskout_probs,
                "origin": ori_probs,
                "mask_sparsity": mask_sparsity,
                "expl_edges": (edge_mask != 0).sum(),
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )

    related_preds = list_to_dict(related_preds)
    related_preds["mask_sparsity"] = related_preds["mask_sparsity"].mean().item()
    related_preds["expl_edges"] = related_preds["expl_edges"].mean().item()
    return related_preds


def eval_related_pred_gc(model, dataset, edge_masks, device, args):
    related_preds = []

    for i in range(len(dataset)):
        data = dataset[i].to(device)
        edge_mask = torch.Tensor(edge_masks[i])
        mask_sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)

        ori_ypred = model(data.x, data.edge_index).cpu().detach().numpy()
        ori_yprob = get_proba(ori_ypred)[0]

        indices = np.where(edge_mask > 0)[0]
        indices_inv = [i for i in range(len(edge_mask)) if i not in indices]

        masked_edge_index = data.edge_index[:, indices].to(device)
        maskout_edge_index = data.edge_index[:, indices_inv].to(device)

        masked_ypred = model(data.x, masked_edge_index).cpu().detach().numpy()
        masked_yprob = get_proba(masked_ypred)[0]

        maskout_ypred = model(data.x, maskout_edge_index).cpu().detach().numpy()
        maskout_yprob = get_proba(maskout_ypred)[0]

        true_label = data.y
        pred_label = np.argmax(ori_yprob)

        related_preds.append(
            {
                "node_idx": -1,
                "masked": masked_yprob,
                "maskout": maskout_yprob,
                "origin": ori_yprob,
                "mask_sparsity": mask_sparsity,
                "expl_edges": (edge_mask != 0).sum(),
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )

    related_preds = list_to_dict(related_preds)
    related_preds["mask_sparsity"] = related_preds["mask_sparsity"].mean().item()
    related_preds["expl_edges"] = related_preds["expl_edges"].mean().item()
    return related_preds


def eval_related_pred_gc_batch(model, dataset, edge_index_set, edge_masks_set, device, args):

    mask_sparsity = get_sparsity(np.hstack(edge_masks_set))
    expl_edges = get_size(np.hstack(edge_masks_set))

    ori_ypred = gnn_preds_gc_batch(model, dataset, edge_index_set, args, device)
    ori_yprob = get_proba(ori_ypred)

    masked_edge_index_set, maskout_edge_index_set = compute_masked_edges_batch(edge_masks_set, edge_index_set, device)

    masked_ypred = gnn_preds_gc_batch(model, dataset, masked_edge_index_set, args, device)
    masked_yprob = get_proba(masked_ypred)

    maskout_ypred = gnn_preds_gc_batch(model, dataset, maskout_edge_index_set, args, device)
    maskout_yprob = get_proba(maskout_ypred)

    related_preds = {
        "masked": masked_yprob,
        "maskout": maskout_yprob,
        "origin": ori_yprob,
        "mask_sparsity": mask_sparsity,
        "expl_edges": expl_edges,
        "true_label": get_true_labels_gc_batch(dataset),
        "pred_label": get_labels(ori_ypred),
    }
    return related_preds


def fidelity_acc(related_preds):
    labels = related_preds["true_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    unimportant_labels = np.argmax(related_preds["maskout"], axis=1)
    p_1 = np.array(ori_labels == labels).astype(int)
    p_2 = np.array(unimportant_labels == labels).astype(int)
    drop_probability = p_1 - p_2
    return drop_probability.mean().item()


def fidelity_acc_inv(related_preds):
    labels = related_preds["true_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    important_labels = np.argmax(related_preds["masked"], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = p_1 - p_2
    return drop_probability.mean().item()


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_prob(related_preds):
    labels = related_preds["true_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    unimportant_probs = np.choose(labels, related_preds["maskout"].T)
    drop_probability = ori_probs - unimportant_probs
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_prob_inv(related_preds):
    labels = related_preds["true_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked"].T)
    drop_probability = ori_probs - important_probs
    return drop_probability.mean().item()


def eval_fidelity(related_preds):
    fidelity_scores = {
        "fidelity_acc+": fidelity_acc(related_preds),
        "fidelity_acc-": fidelity_acc_inv(related_preds),
        "fidelity_prob+": fidelity_prob(related_preds),
        "fidelity_prob-": fidelity_prob_inv(related_preds),
        "mask_sparsity": related_preds["mask_sparsity"],
        "expl_edges": related_preds["expl_edges"],
    }
    return fidelity_scores
