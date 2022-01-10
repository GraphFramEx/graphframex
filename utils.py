import os
import pandas as pd
import numpy as np

import torch
from torch_geometric.utils import k_hop_subgraph

def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def list_to_dict(preds):
    preds_dict = pd.DataFrame(preds).to_dict('list')
    for key in preds_dict.keys():
        preds_dict[key] = np.array(preds_dict[key])
    return(preds_dict)

def normalize(x):
    return (x-min(x))/(max(x)-min(x))

def normalize_masks(edge_masks):
    new_list = []
    for mask in edge_masks:
        new_list.append(normalize(mask))
    return new_list

def get_subgraph(node_idx, x, edge_index, num_hops, **kwargs):
    num_nodes, num_edges = x.size(0), edge_index.size(1)

    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True,
        num_nodes=num_nodes)

    x = x[subset]
    for key, item in kwargs.items():
        if torch.is_tensor(item) and item.size(0) == num_nodes:
            item = item[subset]
        elif torch.is_tensor(item) and item.size(0) == num_edges:
            item = item[edge_mask]
        kwargs[key] = item

    return x, edge_index, mapping, edge_mask, subset, kwargs
