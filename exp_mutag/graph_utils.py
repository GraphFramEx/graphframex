import torch
import numpy as np
from torch.autograd import Variable
from code.utils.gen_utils import from_adj_to_edge_index


def get_edge_index_set(dataset):
    edge_index_set = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False)
        edge_index = []
        for a in adj:
            edge_index.append(from_adj_to_edge_index(a))
        edge_index_set.append(edge_index)
    return edge_index_set


def compute_masked_edges(edge_masks_set, edge_index_set, device):
    
    masked_edge_index_set = []
    maskout_edge_index_set = []

    for batch_idx, edge_masks in enumerate(edge_masks_set):
        edge_index = edge_index_set[batch_idx]
        masked_edge_index = []
        maskout_edge_index = []
        
        for i in range(len(edge_masks)):
            edge_mask = torch.Tensor(edge_masks[i])
            indices = (np.where(edge_mask > 0)[0]).astype('int')
            indices_inv = [i for i in range(len(edge_mask)) if i not in indices]
            masked_edge_index.append(edge_index[i][:, indices].to(device))
            maskout_edge_index.append(edge_index[i][:, indices_inv].to(device))

        masked_edge_index_set.append(masked_edge_index)
        maskout_edge_index_set.append(maskout_edge_index)
    
    return masked_edge_index_set, maskout_edge_index_set