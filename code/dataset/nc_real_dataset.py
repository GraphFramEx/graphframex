import os
import os.path as osp
import torch
import numpy as np
import scipy.sparse as sp
import shutil
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from sklearn.model_selection import train_test_split


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def from_edge_index_to_sparse_adj(edge_index, edge_attr, max_n):
    adj = sp.coo_matrix(
        (edge_attr, (edge_index[0, :], edge_index[1, :])),
        shape=(max_n, max_n),
        dtype=np.float32,
    )
    return adj


def from_sparse_adj_to_edge_index(adj):
    adj = adj.tocoo().astype(np.float32)
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    edge_attr = torch.from_numpy(adj.data)
    return edge_index, edge_attr


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class NCRealGraphDataset:
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset
    """
    names = {
        "facebook": ["Facebook", "FacebookPagePage"],
        "cora": ["Cora", "Planetoid"],
        "citeseer": ["CiteSeer", "Planetoid"],
        "pubmed": ["PubMed", "Planetoid"],
        "chameleon": ["Chameleon", "WikipediaNetwork"],
        "squirrel": ["Squirrel", "WikipediaNetwork"],
        "actor": ["Actor", "Actor"],
        "texas": ["Texas", "WebKB"],
        "cornell": ["Cornell", "WebKB"],
        "wisconsin": ["Wisconsin", "WebKB"],
    }

    def __init__(self, root, name):
        self.root = root
        self.name = name.lower()
        self.data = None
        assert self.name in self.names.keys()
        self.data_save_dir = osp.join(self.root, self.name)
        self.origin_dir = osp.join(self.root, self.raw_name())

    def group_name(self):
        return self.names[self.name][1]

    def raw_name(self):
        return self.names[self.name][0]

    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    def processed_file_names(self):
        return osp.join(self.processed_dir(), "data.pt")

    def download(self):
        """Download the data if it does not exist."""
        self.group = self.group_name()
        self.raw_name = self.raw_name()
        if self.group == "Planetoid":
            Planetoid(self.root, name=self.raw_name)
            os.rename(self.origin_dir, self.data_save_dir)
        elif self.group == "WebKB":
            WebKB(self.data_save_dir, name=self.raw_name)
        elif self.group == "WikipediaNetwork":
            WikipediaNetwork(self.data_save_dir, name=self.name)
            origin_dir = osp.join(self.data_save_dir, "geom_gcn")
            # fetch all files
            for folder_name in os.listdir(origin_dir):
                # construct full file path
                source = osp.join(self.origin_dir, folder_name)
                destination = osp.join(self.data_save_dir, folder_name)
                # move only folder
                print(f"Moving {source} to {destination}")
                if osp.isdir(source):
                    print("moving folder {} to {}".format(source, destination))
                    shutil.move(source, destination)
            shutil.rmtree(origin_dir, ignore_errors=True)
        else:
            eval(self.group)(self.data_save_dir)

    def process(self):
        """Preprocess the data for real dataset by defining a Pytorch geometric data object."""
        if not os.path.isfile(self.processed_file_names()):
            self.download()
        data, _ = torch.load(self.processed_file_names())
        adj = from_edge_index_to_sparse_adj(
            data.edge_index,
            np.ones(data.edge_index.shape[1]),
            data.num_nodes,
        )
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = adj.tocoo().astype(np.float32)

        data.x = torch.FloatTensor(np.array(data.x))
        data.y = torch.LongTensor(data.y)
        data.edge_index, data.edge_attr = from_sparse_adj_to_edge_index(adj)

        if self.name == "facebook":
            n = data.num_nodes
            data.train_mask, data.val_mask, data.test_mask = (
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
            )
            train_ids, test_ids = train_test_split(
                range(n),
                test_size=self.dataset_params["test_ratio"],
                random_state=self.dataset_params["seed"],
                shuffle=True,
            )
            train_ids, val_ids = train_test_split(
                train_ids,
                test_size=self.dataset_params["val_ratio"],
                random_state=self.dataset_params["seed"],
                shuffle=True,
            )
            data.train_mask[train_ids] = 1
            data.val_mask[val_ids] = 1
            data.test_mask[test_ids] = 1

        if data.train_mask.dim() > 1:
            k = self.dataset_params["seed"] % 10
            data.train_mask = data.train_mask[:, k]
            data.val_mask = data.val_mask[:, k]
            data.test_mask = data.test_mask[:, k]

        if data.edge_attr is None:
            data.edge_attr = torch.ones(
                data.edge_index.size(1), device=data.x.device, requires_grad=True
            )
        self.data = data
        return [data]

    def __repr__(self):
        return "{}".format(self.names[self.name][0])
