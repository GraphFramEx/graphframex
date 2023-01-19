""" fidelity.py
    Compute Fidelity+ and Fidelity- scores.
"""

import numpy as np
from sympy import re
import torch
from utils.gen_utils import list_to_dict, get_proba


def eval_related_pred_nc(
    model, data, edge_masks, node_feat_masks, list_node_idx, device, args
):
    """Evaluate related predictions for a single node.

    Args:
        model: trained GNN model
        data: initial data object
        edge_masks: edge masks for the testing node
        node_feat_masks: node features masks for the testing node
        list_node_idx: list of testing nodes

    Returns:
        related_pred: dictionary of related predictions with masked and maskout predictions
    """
    related_preds = []
    data = data.to(device)
    ori_ypred = (
        model(data.x, data.edge_index, edge_weight=data.edge_weight)
        .cpu()
        .detach()
        .numpy()
    )
    ori_yprob = get_proba(ori_ypred)

    num_test = args.num_test_final if args.E else args.num_test

    for i in range(num_test):

        if not args.NF:
            x_masked = data.x
            x_maskout = data.x
        else:
            node_feat_mask = torch.Tensor(node_feat_masks[i]).to(device)
            if node_feat_mask.dim() == 2:
                x_masked = node_feat_mask
                x_maskout = 1 - node_feat_mask
            else:
                x_masked = data.x * node_feat_mask
                x_maskout = data.x * (1 - node_feat_mask)

        if not args.E:
            if eval(args.hard_mask):
                masked_ypred = model(x_masked, data.edge_index).cpu().detach().numpy()
                maskout_ypred = model(x_maskout, data.edge_index).cpu().detach().numpy()
            else:
                masked_ypred = (
                    model(x_masked, data.edge_index, edge_weight=data.edge_weight)
                    .cpu()
                    .detach()
                    .numpy()
                )
                maskout_ypred = (
                    model(x_maskout, data.edge_index, edge_weight=data.edge_weight)
                    .cpu()
                    .detach()
                    .numpy()
                )

        else:
            edge_mask = torch.Tensor(edge_masks[i]).to(device)
            if eval(args.hard_mask):
                masked_edge_index = data.edge_index[:, edge_mask > 0].to(device)
                maskout_edge_index = data.edge_index[:, edge_mask <= 0].to(device)
                masked_ypred = model(x_masked, masked_edge_index).cpu().detach().numpy()
                maskout_ypred = (
                    model(x_maskout, maskout_edge_index).cpu().detach().numpy()
                )
            else:
                masked_ypred = (
                    model(
                        x_masked,
                        data.edge_index,
                        edge_weight=data.edge_weight * edge_mask,
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
                maskout_ypred = (
                    model(
                        x_maskout,
                        data.edge_index,
                        edge_weight=data.edge_weight * (1 - edge_mask),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            edge_mask = edge_mask.cpu().detach().numpy()

        node_idx = list_node_idx[i]
        masked_yprob = get_proba(masked_ypred)
        maskout_yprob = get_proba(maskout_ypred)

        ori_probs = ori_yprob[node_idx]
        masked_probs = masked_yprob[node_idx]
        maskout_probs = maskout_yprob[node_idx]
        true_label = data.y[node_idx].cpu().numpy()
        pred_label = np.argmax(ori_probs)

        # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."\
        related_preds.append(
            {
                "node_idx": node_idx,
                "masked": masked_probs,
                "maskout": maskout_probs,
                "origin": ori_probs,
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )

    related_preds = list_to_dict(related_preds)
    return related_preds


def fidelity_acc(related_preds):
    labels = related_preds["true_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    unimportant_labels = np.argmax(related_preds["maskout"], axis=1)
    p_1 = np.array(ori_labels == labels).astype(int)
    p_2 = np.array(unimportant_labels == labels).astype(int)
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


def fidelity_acc_inv(related_preds):
    labels = related_preds["true_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    important_labels = np.argmax(related_preds["masked"], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


def fidelity_gnn_acc(related_preds):
    labels = related_preds["pred_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    unimportant_labels = np.argmax(related_preds["maskout"], axis=1)
    p_1 = np.array(ori_labels == labels).astype(int)
    p_2 = np.array(unimportant_labels == labels).astype(int)
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


def fidelity_gnn_acc_inv(related_preds):
    labels = related_preds["pred_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    important_labels = np.argmax(related_preds["masked"], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = np.abs(p_1 - p_2)
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


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_gnn_prob(related_preds):
    labels = related_preds["pred_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    unimportant_probs = np.choose(labels, related_preds["maskout"].T)
    drop_probability = np.abs(ori_probs - unimportant_probs)
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_gnn_prob_inv(related_preds):
    labels = related_preds["pred_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked"].T)
    drop_probability = np.abs(ori_probs - important_probs)
    return drop_probability.mean().item()


def eval_fidelity(related_preds, args):
    if eval(args.true_label_as_target):
        fidelity_scores = {
            "fidelity_acc+": fidelity_acc(related_preds),
            "fidelity_acc-": fidelity_acc_inv(related_preds),
            "fidelity_prob+": fidelity_prob(related_preds),
            "fidelity_prob-": fidelity_prob_inv(related_preds),
        }
    else:
        fidelity_scores = {
            "fidelity_gnn_acc+": fidelity_gnn_acc(related_preds),
            "fidelity_gnn_acc-": fidelity_gnn_acc_inv(related_preds),
            "fidelity_gnn_prob+": fidelity_gnn_prob(related_preds),
            "fidelity_gnn_prob-": fidelity_gnn_prob_inv(related_preds),
        }

    return fidelity_scores
