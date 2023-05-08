import numpy as np
import json
from scipy.stats import entropy, gaussian_kde
import matplotlib.pyplot as plt
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data


def topk_edges_unique(edge_mask, edge_index, num_top_edges):
    """Return the indices of the top-k edges in the mask.

    Args:
        edge_mask (Tensor): edge mask of shape (num_edges,).
        edge_index (Tensor): edge index tensor of shape (2, num_edges)
        num_top_edges (int): number of top edges to be kept
    """
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
    if len(x) > 0 and not np.all(np.isnan(x)):
        if (np.nanmax(x) - np.nanmin(x)) == 0:
            return x
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    else:
        return x


def normalize_all_masks(masks):
    for i in range(len(masks)):
        masks[i] = normalize_mask(masks[i])
    return masks


def clean(masks):
    """Clean masks by removing NaN, inf and too small values and normalizing"""
    for i in range(len(masks)):
        if (masks[i] is not None) and len(masks[i]) > 0:
            masks[i] = np.nan_to_num(masks[i], copy=True, nan=0.0, posinf=10, neginf=-10)
            masks[i] = np.clip(masks[i], -10, 10)
            masks[i] = normalize_mask(masks[i])
            masks[i] = np.where(masks[i] < 0.01, 0, masks[i])
    return masks


def clean_all_masks(edge_masks, node_feat_masks, args):
    if args.E:
        ### Mask normalisation and cleaning ###
        edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
        edge_masks = clean(edge_masks)
    if args.NF:
        ### Mask normalisation and cleaning ###
        node_feat_masks = [
            node_feat_mask.astype("float") for node_feat_mask in node_feat_masks
        ]
        node_feat_masks = clean(node_feat_masks)
    return edge_masks, node_feat_masks


def from_mask_to_nxsubgraph(mask, node_index, data):
    """Convert mask to a networkx subgraph."""
    masked_edge_index = data.edge_index[:, mask > 0]
    weights = mask[mask > 0]
    if masked_edge_index.size(1) == 0:
        return None
    masked_edge_index = masked_edge_index.cpu().numpy()

    lst = np.sort(np.unique(masked_edge_index))
    if node_index not in lst:
        return None
    # lst = np.delete(lst, np.where(lst == node_index))
    # lst = np.insert(lst, 0, node_index)
    # d = {lst[i]: i for i in range(0,len(lst))}
    relabeled_node_index = np.where(lst == node_index)[0][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masked_edge_index = torch.LongTensor(masked_edge_index).to(device)
    lst = torch.LongTensor(lst).to(device)
    sub_edge_index, sub_edge_attr, sub_mask = subgraph(
        lst, masked_edge_index, weights, relabel_nodes=True, return_edge_mask=True
    )
    masked_x = data.x[lst]
    data = Data(x=masked_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
    data["weight"] = sub_edge_attr
    G = to_networkx(data, edge_attrs=["weight"])
    return G, relabeled_node_index


def get_ratio_connected_components(edge_masks, edge_index):
    """Compute connected components ratio of the edge mask."""
    cc_ratio = []
    for i in range(len(edge_masks)):
        edge_mask = edge_masks[i]
        masked_edge_index = edge_index[:, edge_mask > 0]
        if masked_edge_index.size(1) == 0:
            return None
        masked_edge_index = masked_edge_index.cpu().numpy()
        lst = np.sort(np.unique(masked_edge_index))
        d = {lst[i]: i for i in range(len(lst))}
        indexer = np.array(
            [
                d.get(i, -1)
                for i in range(masked_edge_index.min(), masked_edge_index.max() + 1)
            ]
        )
        relabel_masked_edge_index = indexer[
            (masked_edge_index - masked_edge_index.min())
        ]
        sparse_adj = to_scipy_sparse_matrix(
            torch.LongTensor(relabel_masked_edge_index)
        ).toarray()
        n_components, labels = connected_components(
            csgraph=sparse_adj, directed=False, return_labels=True
        )
        cc_ratio.append(n_components / len(labels))
    return np.mean(cc_ratio)


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
        # kde = gaussian_kde(np.array(pos_mask))
        # density = kde(pos_mask)
        # index = np.argmax(density)
        ys, xs, _ = plt.hist(pos_mask, bins=100)
        index = np.argmax(ys)
        max_avg += xs[index]
        k += 1
    if k == 0:
        return -1
    return max_avg / k


def get_mask_properties(masks):
    mask_info = {
        "mask_size": get_size(masks),
        "mask_sparsity": get_sparsity(masks),
        "mask_entropy": get_entropy(masks),
        "max_avg": get_avg_max(masks),
    }
    return mask_info


# def get_explanatory_subgraph__properties():

# Edge_masks are normalized; we then select only the edges for which the mask value > threshold
# transform edge_mask:
# normalisation
# sparsity
# hard or soft


def transform_mask(masks, data, param, args):
    """Transform masks according to the given strategy (topk, threshold, sparsity) and level."""
    new_masks = []
    for mask_ori in masks:
        mask = mask_ori.copy()
        if args.strategy == "topk":
            if eval(args.directed):
                unimportant_indices = (-mask).argsort()[param:]
                mask[unimportant_indices] = 0
            else:
                mask = mask_to_shape(mask, data.edge_index, param)
                # indices = np.where(mask > 0)[0]
        if args.strategy == "sparsity":
            mask = control_sparsity(mask, param)
        if args.strategy == "threshold":
            mask = np.where(mask > param, mask, 0)
        new_masks.append(mask)
    return np.array(new_masks, dtype=np.float64)


def mask_to_shape(mask, edge_index, num_top_edges):
    """Modify the mask by selecting only the num_top_edges edges with the highest mask value."""
    indices = topk_edges_unique(mask, edge_index, num_top_edges)
    unimportant_indices = [i for i in range(len(mask)) if i not in indices]
    new_mask = mask.copy()
    new_mask[unimportant_indices] = 0
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
