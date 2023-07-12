""" accountability.py
    Compute Accountability scores and the extended explanation.
"""

import numpy as np
import torch


def extend_mask(mask, mu=0.1):
    """Extend the explanation by adding the remaining entities."""
    # return np.where(mask == 0, mu, mask)
    return mask + mu


# Accountability metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Higher accountability value indicates good explanations -->1
# Interval [-1;1]
def accountability(related_preds):
    labels = related_preds["true_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked"].T)
    drop_probability = important_probs - ori_probs
    return drop_probability


def accountability_gnn(related_preds):
    labels = related_preds["pred_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked"].T)
    drop_probability = important_probs - ori_probs
    return drop_probability


def accountability_ext(related_preds):
    labels = related_preds["true_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked_extended"].T)
    drop_probability = important_probs - ori_probs
    return drop_probability


def accountability_gnn_ext(related_preds):
    labels = related_preds["pred_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked_extended"].T)
    drop_probability = important_probs - ori_probs
    return drop_probability
