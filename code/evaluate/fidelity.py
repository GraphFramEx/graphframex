""" fidelity.py
    Compute Fidelity+ and Fidelity- scores.
"""

import numpy as np


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
