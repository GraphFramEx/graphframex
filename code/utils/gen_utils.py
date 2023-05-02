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
    to_dense_adj,
    subgraph,
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

def padded_datalist(data_list, adj_list, max_num_nodes):
    for i, data in enumerate(data_list):
        data.adj_padded = padding_graphs(adj_list[i], max_num_nodes)
        data.x_padded = padding_features(data.x, max_num_nodes)
    return data_list

def padding_graphs(adj, max_num_nodes):
    num_nodes = adj.shape[0]
    adj_padded = np.eye((max_num_nodes))
    adj_padded[:num_nodes, :num_nodes] = adj.cpu()
    return torch.tensor(adj_padded, dtype=torch.long)

def padding_features(features, max_num_nodes):
    feat_dim = features.shape[1]
    num_nodes = features.shape[0]
    features_padded = np.zeros((max_num_nodes, feat_dim))
    features_padded[:num_nodes] = features.cpu()
    return torch.tensor(features_padded, dtype=torch.float)


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


def get_subgraph(
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


def from_edge_index_to_adj_torch(edge_index, edge_weight, max_n):
    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(max_n, max_n),
        dtype=torch.float32,
        device=edge_index.device,
    )
    if edge_index.requires_grad:
        adj.requires_grad = True
    return adj.to_dense()


def from_sparse_adj_to_edge_index(adj):
    adj = adj.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    edge_weight = torch.from_numpy(adj.data.astype(np.float32))
    return edge_index, edge_weight


def from_adj_to_edge_index_torch(adj):
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    # if adj.requires_grad:
    # edge_index.requires_grad = True
    # edge_attr.requires_grad = True
    return edge_index, edge_attr


def convert_coo_to_tensor(adj):
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


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


def normalize_adj(adj):
    # Normalize adjacancy matrix according to reparam trick in GCN paper
    A_tilde = adj + torch.eye(adj.shape[0])
    D_tilde = get_degree_matrix(A_tilde)
    # Raise to power -1/2, set all infs to 0s
    D_tilde_exp = D_tilde ** (-1 / 2)
    D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

    # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
    norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
    return norm_adj


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
    edge_subset_relabel = subgraph(
        edge_subset[0], edge_index, edge_attr=None, relabel_nodes=True
    )  # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sub_feat = features[edge_subset[0], :]
    sub_labels = labels[edge_subset[0]]
    new_index = np.array([i for i in range(len(edge_subset[0]))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))  # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    return sub_adj, sub_feat, sub_labels, node_dict


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector

def get_cmn_edges(new_edge_index, new_edge_weight, edge_index):
    cmn_init_edge_idx, cmn_edges = [], []
    cmn_edge_weight = np.zeros(edge_index.shape[1])
    list_tuples = zip(*new_edge_index.cpu().numpy())
    for d, (u, v) in enumerate(list_tuples):
        pos_new_edge = np.where(np.all(np.array(edge_index.T)==(u,v),axis=1))[0]
        if len(pos_new_edge) > 0:
            idx_init = pos_new_edge[0]
            cmn_init_edge_idx.append(idx_init)
            cmn_edges.append([u, v])
            cmn_edge_weight[idx_init] = new_edge_weight[d] 
            #idx_init is the index of the edge in the initial edges that exists in the new edges.
    return cmn_init_edge_idx, cmn_edges, cmn_edge_weight


def filter_existing_edges(perturb_edges, edge_index):
    """Returns a mask of edges that are perturbed by CF-GNNExplainer and also exist in the original graph
    Args:
        perturb_edges (_type_): counterfactual explanations, i.e. edges that are perturbed by CF-GNNExplainer
        edge_index (_type_): edge index of the original graph
    Returns:
        _type_: edge mask with value 1 if the edge exists.
    """
    edge_mask = np.zeros(edge_index.shape[1])
    list_tuples = zip(*perturb_edges)
    for i in range(edge_index.shape[1]):
        # if include_edges is not None and not include_edges[i].item():
        # continue
        u, v = list(edge_index[:, i])
        if (u, v) in list_tuples:
            edge_mask[i] = 1
    return edge_mask    


def get_existing_edges(new_edge_index, new_edge_weight, edge_index, edge_attr):
    keep_edge_idx = []
    for i in range(len(new_edge_index.T)):
        elmt = np.array(new_edge_index.T[i])
        pos_new_edge = np.where(np.all(np.array(edge_index.T)==elmt,axis=1))[0]
        if pos_new_edge.size > 0:
            keep_edge_idx.append(pos_new_edge[0])
    kept_edges = edge_index.T[keep_edge_idx]
    kept_edges = np.array(kept_edges)
    kept_edge_attr = edge_attr[keep_edge_idx]
    kept_edge_weight = new_edge_weight[keep_edge_idx]
    if kept_edges.ndim == 1:
        kept_edges = kept_edges.reshape(0,2)
    return(keep_edge_idx, kept_edges, kept_edge_attr, kept_edge_weight)


def get_new_edges(new_edge_index, new_edge_weight, edge_index, edge_attr):
    new_added_edges = []
    new_added_edge_idx = []
    for i in range(len(new_edge_index.T)):
        elmt = np.array(new_edge_index.T[i])
        pos_new_edge = np.where(np.all(np.array(edge_index.T)==elmt,axis=1))[0]
        if pos_new_edge.size == 0:
            new_added_edges.append(elmt)
            new_added_edge_idx.append(i)
    new_added_edges = np.array(new_added_edges)
    mean_feat = np.mean(np.array(edge_attr),0)
    var_feat = np.var(np.array(edge_attr),0)
    new_added_edge_attr= np.array([np.random.normal(loc=mean_feat[i], scale=var_feat[i], size=(len(new_added_edges))) for i in range(edge_attr.shape[1])]).T
    new_added_edge_weight = new_edge_weight[new_added_edge_idx]
    if new_added_edges.ndim == 1:
        new_added_edges = new_added_edges.reshape(0,2)
    return(new_added_edge_idx, new_added_edges, new_added_edge_attr, new_added_edge_weight)


def get_cf_edge_mask(new_edge_index, edge_index):
    cmn_edge_idx = []
    edge_mask = torch.ones(edge_index.shape[1])
    new_edges = new_edge_index.T.cpu().numpy()
    existing_edges = edge_index.T.cpu().numpy()
    for i in range(len(new_edges)):
        elmt = np.array(new_edges[i])
        pos_new_edge = np.where(np.all(existing_edges==elmt,axis=1))[0]
        if pos_new_edge.size > 0:
            cmn_edge_idx.append(pos_new_edge[0])
    # The explanation is the edges that are not counterfactual edges
    edge_mask[cmn_edge_idx] = 0
    return(edge_mask)


