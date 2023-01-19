import random

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from scipy.special import softmax
from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    k_hop_subgraph,
    to_scipy_sparse_matrix,
)


def list_to_dict(preds):
    preds_dict = pd.DataFrame(preds).to_dict("list")
    for key in preds_dict.keys():
        preds_dict[key] = np.array(preds_dict[key])
    return preds_dict


def sample_large_graph(data):
    if data.num_edges > 50000:
        print("Too many edges, sampling large graph...")
        node_idx = random.randint(0, data.num_nodes - 1)
        x, edge_index, mapping, edge_mask, subset, kwargs = get_subgraph(
            node_idx, data.x, data.edge_index, num_hops=3
        )
        data = data.subgraph(subset)
        print(f"Sample size: {data.num_nodes} nodes and {data.num_edges} edges")
    return data


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


def subgraph(
    node_idx,
    num_hops,
    edge_index,
    relabel_nodes=False,
    num_nodes=None,
    flow="source_to_target",
):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor(
            [node_idx], device=row.device, dtype=torch.int64
        ).flatten()
    else:
        node_idx = node_idx.to(row.device)

    inv = None

    if num_hops != -1:
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[: node_idx.numel()]
    else:
        subsets = node_idx
        cur_subsets = node_idx
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


def from_edge_index_to_adj(edge_index, edge_weight, max_n):
    adj = to_scipy_sparse_matrix(edge_index, edge_attr=edge_weight).toarray()
    assert len(adj) <= max_n, "The adjacency matrix contains more nodes than the graph!"
    if len(adj) < max_n:
        adj = np.pad(adj, (0, max_n - len(adj)), mode="constant")
    return torch.FloatTensor(adj)


def from_adj_to_edge_index(adj):
    A = csr_matrix(adj)
    edges, edge_weight = from_scipy_sparse_matrix(A)
    return edges, edge_weight


def from_edge_index_to_sparse_adj(edge_index, edge_weight, max_n):
    adj = sp.coo_matrix(
        (edge_weight, (edge_index[0, :], edge_index[1, :])),
        shape=(max_n, max_n),
        dtype=np.float32,
    )
    return adj


def from_sparse_adj_to_edge_index(adj):
    adj = adj.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    edge_weight = torch.from_numpy(adj.data)
    return edge_index, edge_weight


def init_weights(edge_index):
    edge_weights = []
    for edges in edge_index:
        edges_w = torch.ones(edges.size(1))
        edge_weights.append(edges_w)
    return edge_weights


def get_test_nodes(data, model, args):
    if args.dataset.startswith(tuple(["ba", "tree"])):
        pred_labels = get_labels(model(data.x, data.edge_index).cpu().detach().numpy())
        if args.testing_pred == "correct":
            list_node_idx = np.where(pred_labels == data.y.cpu().numpy())[0]
        if args.testing_pred == "wrong":
            list_node_idx = np.where(pred_labels != data.y.cpu().numpy())[0]
        else:  # args.testing_pred is "mix"
            list_node_idx = np.arange(data.x.size(0))
        list_node_idx_pattern = list_node_idx[list_node_idx > args.num_basis]
        # list_test_nodes = [x.item() for x in list_node_idx_pattern[: args.num_test]]
        list_test_nodes = [
            x.item()
            for x in np.random.choice(
                list_node_idx_pattern,
                size=min(args.num_test, len(list_node_idx_pattern)),
                replace=False,
            )
        ]
    else:
        pred_labels = get_labels(
            model(data.x, data.edge_index, edge_weight=data.edge_weight)
            .cpu()
            .detach()
            .numpy()
        )
        if args.testing_pred == "correct":
            list_node_idx = np.where(pred_labels == data.y.cpu().numpy())[0]
        if args.testing_pred == "wrong":
            list_node_idx = np.where(pred_labels != data.y.cpu().numpy())[0]
        else:  # args.testing_pred is "mix"
            list_node_idx = np.arange(data.x.size(0))

        #### For ebay graph: only explain fraudulent nodes!! ####
        if args.dataset == "ebay":
            list_node_idx = np.where(pred_labels != 0)[
                0
            ]  ##only fraudulent nodes are kept!

        list_test_nodes = [
            x.item()
            for x in np.random.choice(
                list_node_idx,
                size=min(args.num_test, len(list_node_idx)),
                replace=False,
            )
        ]
    return list_test_nodes


def get_proba(ypred):
    yprob = softmax(ypred, axis=1)
    return yprob


def get_labels(ypred):
    ylabels = np.argmax(ypred, axis=1)
    return ylabels
