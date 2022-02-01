import numpy as np
import torch
from scipy.special import softmax

from gnn_eval import get_proba
from graph_utils import compute_masked_edges


def topk_edges_directed(edge_mask, edge_index, num_top_edges):
    indices = (-edge_mask).argsort()
    top = np.array([], dtype='int')
    i = 0
    list_edges = np.sort(edge_index.cpu().T, axis=1)
    while len(top)<num_top_edges:
        subset = indices[num_top_edges*i:num_top_edges*(i+1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]])
        i+=1
    return top[:num_top_edges]

    

def normalize_mask(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def normalize_all_masks(masks):
    for i in range(len(masks)):
        masks[i] = normalize_mask(masks[i])
    return masks

def clean_masks(masks):
    for i in range(len(masks)):
        masks[i] = np.nan_to_num(masks[i], copy=True, nan=0.0, posinf=10, neginf=-10)
        masks[i] = np.clip(masks[i], -10, 10)
    return masks


def get_sparsity(masks):
    sparsity = 0
    for i in range(len(masks)):
        sparsity += 1.0 - (masks[i]!= 0).sum() / len(masks[i])
    return sparsity/len(masks)

def get_size(masks):
    size = 0
    for i in range(len(masks)):
        size += len(masks[i][masks[i]>0])
    return size/len(masks)
# Edge_masks are normalized; we then select only the edges for which the mask value > threshold
#transform edge_mask:
# normalisation
# sparsity
# hard or soft

def transform_mask(masks, args):
    new_masks = []
    for mask in masks:
        if args.topk >= 0:
            unimportant_indices = (-mask).argsort()[args.topk:]
            mask[unimportant_indices] = 0
        if args.sparsity>=0:
            mask = control_sparsity(mask, args.sparsity)
        if args.threshold>=0:
            mask = np.where(mask>args.threshold, mask, 0)
        new_masks.append(mask)
    return(new_masks)

def mask_to_shape(mask, edge_index, num_top_edges):
    indices = topk_edges_directed(mask, edge_index, num_top_edges)
    new_mask = np.zeros(len(mask))
    new_mask[indices] = 1
    return new_mask

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




##### Fidelity #####

    
def get_true_labels(dataset):
    labels = []
    for batch_idx, data in enumerate(dataset):
        labels.append(data["label"].long().numpy())
    labels = np.hstack(labels)
    return labels
    
def get_proba(ypred):
    yprob = softmax(ypred, axis=1)
    return yprob

def get_labels(ypred):
    ylabels = torch.argmax(ypred, dim=1)
    return ylabels

def list_to_dict(preds):
    preds_dict=pd.DataFrame(preds).to_dict('list')
    for key in preds_dict.keys():
        preds_dict[key] = preds_dict[key][0]
    return(preds_dict)

def eval_related_pred_batch(model, dataset, edge_index_set, edge_masks_set, device, args):
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

    masked_edge_index_set, maskout_edge_index_set = compute_masked_edges(edge_masks_set, edge_index_set, device)

    masked_ypred = gnn_preds_gc(model, dataset, masked_edge_index_set, args)
    masked_yprob = get_proba(masked_ypred)

    maskout_ypred = gnn_preds_gc(model, dataset, maskout_edge_index_set, args)
    maskout_yprob = get_proba(maskout_ypred)

    #true_label = [data["label"].long().numpy() for batch_idx, data in enumerate(dataset)]
    #pred_label = get_labels(ori_ypred)
    #print(true_label)
    #print(pred_label)
    # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

    related_preds = {'masked': masked_yprob,
                          'maskout': maskout_yprob,
                          'origin': ori_yprob,
                          'mask_sparsity': mask_sparsity,
                          'expl_edges': expl_edges, 
                        'true_label': get_true_labels(dataset)
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


