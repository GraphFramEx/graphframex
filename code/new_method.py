import os
import networkx as nx
import numpy as np
import torch

"""def new_explainer(model, data, node_idx, target, device, args):
    ....__annotations__
    return edge_mask, node_feat_mask"""

def new_method(model, data, node_idx, target, device, args):
    edge_mask = np.random.uniform(size=data.edge_index.shape[1])
    node_feat_mask = np.random.uniform(size=data.x.shape[1])
    return edge_mask, node_feat_mask
