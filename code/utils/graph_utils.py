import numpy as np
import torch
from torch.autograd import Variable

from utils.gen_utils import from_adj_to_edge_index


def get_edge_index_set_loader(dataset):
    edge_index_set = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False)
        edge_index = []
        for a in adj:
            edge_index.append(from_adj_to_edge_index(a))
        edge_index_set.append(edge_index)
    return edge_index_set


def get_edge_index_set(dataset):
    edge_index = []
    for data in dataset:
        adj = data["adj"]
        edge_index.append(from_adj_to_edge_index(adj))
    return edge_index


def compute_masked_edges(edge_masks, edge_index, device):

    masked_edge_index = []
    maskout_edge_index = []

    for i in range(len(edge_masks)):
        edge_mask = torch.Tensor(edge_masks[i])
        indices = (np.where(edge_mask > 0)[0]).astype("int")
        indices_inv = [i for i in range(len(edge_mask)) if i not in indices]
        masked_edge_index.append(edge_index[i][:, indices].to(device))
        maskout_edge_index.append(edge_index[i][:, indices_inv].to(device))

    return masked_edge_index, maskout_edge_index
