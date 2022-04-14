import numpy as np
from scipy.stats import entropy, gaussian_kde
import matplotlib.pyplot as plt
import copy


def topk_edges_directed(edge_mask, edge_index, num_top_edges):
    indices = (-edge_mask).argsort()
    top = np.array([], dtype="int")
    i = 0
    list_edges = np.sort(edge_index.cpu().T, axis=1)
    while len(top) < num_top_edges:
        subset = indices[num_top_edges * i : num_top_edges * (i + 1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]])
        i += 1
    return top[:num_top_edges]


def normalize_mask(x):
    if (np.nanmax(x) - np.nanmin(x)) == 0:
        return x
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def normalize_all_masks(masks):
    for i in range(len(masks)):
        masks[i] = normalize_mask(masks[i])
    return masks


def clean_masks(masks):
    for i in range(len(masks)):
        masks[i] = np.nan_to_num(masks[i], copy=True, nan=0.0, posinf=10, neginf=-10)
        masks[i] = np.clip(masks[i], -10, 10)
        masks[i] = normalize_mask(masks[i])
        masks[i] = np.where(masks[i] < 0.001, 0, masks[i])
    print('masks cleaned')
    return masks


def get_sparsity(masks):
    sparsity = 0
    for i in range(len(masks)):
        sparsity += 1.0 - (masks[i] != 0).sum() / len(masks[i])
    return sparsity / len(masks)


def get_size(masks):
    size = 0
    for i in range(len(masks)):
        size += (masks[i] != 0).sum()
    return size / len(masks)

def get_entropy(masks):
    ent = 0
    k = 0
    for i in range(len(masks)):
        pos_mask = masks[i][masks[i] > 0]
        if len(pos_mask) == 0:
            continue
        ent += entropy(pos_mask)
        k += 1
    if k == 0:
        return -1
    return ent / k

def get_avg_max(masks):
    max_avg = 0
    k = 0
    for i in range(len(masks)):
        pos_mask = masks[i][masks[i] > 0]
        if len(pos_mask) == 0:
            continue
        #kde = gaussian_kde(np.array(pos_mask))
        #density = kde(pos_mask) 
        #index = np.argmax(density)
        ys, xs, _ = plt.hist(pos_mask, bins=100)
        index = np.argmax(ys)
        max_avg += xs[index] 
        k += 1
    if k == 0:
        return -1
    return max_avg / k

def get_mask_info(masks):
    mask_info = {'mask_size': get_size(masks), 'mask_entropy': get_entropy(masks), 'max_avg': get_avg_max(masks)}
    return mask_info

# Edge_masks are normalized; we then select only the edges for which the mask value > threshold
# transform edge_mask:
# normalisation
# sparsity
# hard or soft


def transform_mask(masks, args):
    new_masks = []
    for mask_ori in masks:
        mask = mask_ori.copy()
        if args.topk >= 0:
            unimportant_indices = (-mask).argsort()[args.topk :]
            mask[unimportant_indices] = 0
        if args.sparsity >= 0:
            mask = control_sparsity(mask, args.sparsity)
        if args.threshold >= 0:
            mask = np.where(mask > args.threshold, mask, 0)
        new_masks.append(mask)
    return np.array(new_masks, dtype=np.float64)


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
