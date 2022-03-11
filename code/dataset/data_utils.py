import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


def split_data(data, args):
    n = data.num_nodes
    data.train_mask, data.val_mask, data.test_mask = (
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
    )
    train_ids, test_ids = train_test_split(range(n), test_size=args.test_ratio, random_state=args.seed, shuffle=True)
    train_ids, val_ids = train_test_split(train_ids, test_size=args.val_ratio, random_state=args.seed, shuffle=True)

    data.train_mask[train_ids] = 1
    data.val_mask[val_ids] = 1
    data.test_mask[test_ids] = 1

    return data


def preprocess_planetoid(data):
    edges = data.edge_index.T
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(data.num_nodes, data.num_nodes), dtype=np.float32
    )
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = data.x
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(data.y)
    data.x = features

    sparse_mx = adj.tocoo().astype(np.float32)
    data.edge_index = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    data.edge_weight = torch.from_numpy(sparse_mx.data)

    return data


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
