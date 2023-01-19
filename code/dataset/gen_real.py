from utils.gen_utils import (
    from_edge_index_to_sparse_adj,
    from_sparse_adj_to_edge_index
)
import numpy as np
import scipy.sparse as sp
import os
import shutil
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from utils.io_utils import check_dir
from dataset.data_utils import get_split, split_data

REAL_DATA = {"facebook": "FacebookPagePage", "cora": "Planetoid", "citeseer": "Planetoid", "pubmed": "Planetoid",
                "chameleon": "WikipediaNetwork", "squirrel": "WikipediaNetwork", 
                "ppi": "PPI", "actor": "Actor", 
                "texas": "WebKB", "cornell": "WebKB", "wisconsin": "WebKB"}
PLANETOIDS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
WEBKB = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}



def load_data_real(args, device):
    check_dir(args.data_save_dir)
    data_dir = os.path.join(args.data_save_dir, args.dataset)
    check_dir(data_dir)
    data_filename = f"{data_dir}/processed/data.pt"
    if not os.path.isfile(data_filename):
        if REAL_DATA[args.dataset] == "Planetoid":
            Planetoid(args.data_save_dir, name=PLANETOIDS[args.dataset])
            origin_dir = os.path.join(args.data_save_dir, PLANETOIDS[args.dataset])
            os.rename(origin_dir, data_dir)
        elif REAL_DATA[args.dataset] == "WebKB":
            WebKB(args.data_save_dir, name=WEBKB[args.dataset])
        elif REAL_DATA[args.dataset] == "WikipediaNetwork":
            WikipediaNetwork(args.data_save_dir, name=args.dataset)
            origin_dir = os.path.join(data_dir, "geom_gcn")
            # fetch all files
            for folder_name in os.listdir(origin_dir):
                # construct full file path
                source =  os.path.join(origin_dir, folder_name)
                destination = os.path.join(data_dir, folder_name)
                # move only folder
                print(f"Moving {source} to {destination}")
                if os.path.isdir(source):
                    print('moving folder {} to {}'.format(source, destination))
                    shutil.move(source, destination)
            shutil.rmtree(origin_dir, ignore_errors=True)
        else:
            eval(REAL_DATA[args.dataset])(data_dir)
    data, _ = torch.load(data_filename)
    data = preprocess_real(data)

    if args.dataset == "facebook":
        data = split_data(data, args)
    if data.train_mask.dim() > 1:
        data = get_split(data, args)
    data = data.to(device)
    if data.edge_weight is None:
        data.edge_weight = torch.ones(data.edge_index.size(1), device=data.x.device, requires_grad=True)
    return data


def preprocess_real(data):
    """ Preprocess the data for real dataset by defining a Pytorch geometric data object."""
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
