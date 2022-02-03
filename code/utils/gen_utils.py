import random

import numpy as np
import pandas as pd
import torch
from dataset.mutag_utils import GraphSampler
from scipy.sparse import csr_matrix
from scipy.special import softmax
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


def get_test_nodes(data, model, args):
    if args.true_label:
        pred_labels = get_labels(model(data.x, data.edge_index).cpu().detach().numpy())
        list_node_idx = np.where(pred_labels == data.y.cpu().numpy())[0]
    else:
        list_node_idx = np.arange(data.x.size(0))
    list_node_idx_pattern = list_node_idx[list_node_idx > args.num_basis]
    list_test_nodes = [x.item() for x in random.choices(list_node_idx_pattern, k=args.num_test)]
    return list_test_nodes


def get_test_graphs(data, args):
    list_test_idx = np.random.randint(0, len(data), args.num_test)
    test_graphs = gen_dataloader(np.array(data)[list_test_idx], args)
    return test_graphs


def gen_dataloader(graphs, args, max_nodes=0):
    dataset_sampler = GraphSampler(
        graphs,
        normalize=False,
        max_num_nodes=max_nodes,
        features=args.feature_type,
    )
    dataset_loader = torch.utils.data.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return dataset_loader


def get_true_labels_gc(dataset):
    labels = []
    for batch_idx, data in enumerate(dataset):
        labels.append(data["label"].long().numpy())
    labels = np.hstack(labels)
    return labels


def get_proba(ypred):
    yprob = softmax(ypred, axis=1)
    return yprob


def get_labels(ypred):
    ylabels = np.argmax(ypred, axis=1)
    return ylabels
