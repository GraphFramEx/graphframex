import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph, to_scipy_sparse_matrix


def list_to_dict(preds):
    preds_dict = pd.DataFrame(preds).to_dict("list")
    for key in preds_dict.keys():
        preds_dict[key] = np.array(preds_dict[key])
    return preds_dict


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def normalize_masks(edge_masks):
    new_list = []
    for mask in edge_masks:
        new_list.append(normalize(mask))
    return new_list


def get_subgraph(node_idx, x, edge_index, num_hops, **kwargs):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
    )

    x = x[subset]
    for key, item in kwargs.items():
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]
        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, subset, kwargs


def from_edge_index_to_adj(edge_index, max_n):
    adj = to_scipy_sparse_matrix(edge_index).toarray()
    assert len(adj) <= max_n, "The adjacency matrix contains more nodes than the graph!"
    if len(adj) < max_n:
        adj = np.pad(adj, (0, max_n - len(adj)), mode="constant")
    return torch.FloatTensor(adj)


def from_adj_to_edge_index(adj):
    A = csr_matrix(adj)
    edges, _ = from_scipy_sparse_matrix(A)
    return torch.LongTensor(edges)
