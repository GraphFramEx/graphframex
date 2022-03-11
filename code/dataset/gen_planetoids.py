from utils.gen_utils import (
    from_adj_to_edge_index,
    from_edge_index_to_adj,
    from_edge_index_to_sparse_adj,
    from_sparse_adj_to_edge_index,
    init_weights,
)

import numpy as np
import scipy.sparse as sp
import torch

PLANETOIDS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def load_data_planetoids(data_filename):
    data, _ = torch.load(data_filename)
    data = preprocess_planetoid(data)
    return data


def preprocess_planetoid(data):
    adj = from_edge_index_to_sparse_adj(data.edge_index, np.ones(data.edge_index.shape[1]), data.num_nodes)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = adj.tocoo().astype(np.float32)

    data.x = torch.FloatTensor(np.array(data.x))
    data.y = torch.LongTensor(data.y)
    data.edge_index, data.edge_weight = from_sparse_adj_to_edge_index(adj)
    return data


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
