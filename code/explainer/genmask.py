import time

import torch
from torch.autograd import Variable
from utils.graph_utils import get_edge_index_batch

from explainer.graph_explainer import *
from explainer.node_explainer import *


def compute_edge_masks_nc(list_test_nodes, model, data, device, args):
    explain_function = eval("explain_" + args.explainer_name + "_node")
    Time = []
    edge_masks = []
    targets = data.y
    for node_idx in list_test_nodes:
        x = torch.FloatTensor(data.x.cpu().numpy().copy()).to(device)
        edge_index = torch.LongTensor(data.edge_index.cpu().numpy().copy()).to(device)
        start_time = time.time()
        edge_mask = explain_function(
            model, node_idx, x, edge_index, targets[node_idx], device, args, edge_weight=data.edge_weight
        )
        end_time = time.time()
        duration_seconds = end_time - start_time
        Time.append(duration_seconds)
        edge_masks.append(edge_mask)
    return edge_masks, Time


def compute_edge_masks_gc(model, test_data, device, args):
    explain_function = eval("explain_" + args.explainer_name + "_graph")
    Time = []
    edge_masks = []

    for i in range(len(test_data)):
        data = test_data[i]
        x = torch.FloatTensor(data.x.cpu().numpy().copy()).to(device)
        edge_index = torch.LongTensor(data.edge_index.cpu().numpy().copy()).to(device)
        start_time = time.time()
        edge_mask = explain_function(model, x, edge_index, data.y, device, args)
        end_time = time.time()
        duration_seconds = end_time - start_time
        Time.append(duration_seconds)
        edge_masks.append(edge_mask)
    return (edge_masks, Time)


def compute_edge_masks_gc_batch(model, test_graphs, device, args):
    explain_function = eval("explain_" + args.explainer_name + "_graph")
    edge_index_set = get_edge_index_batch(test_graphs)
    edge_masks_set = []
    Time = []

    for batch_idx, data in enumerate(test_graphs):
        edge_masks = []

        h0 = Variable(data["feats"].float()).to(device)
        targets = data["label"].long().numpy()

        for i in range(len(edge_index_set[batch_idx])):
            start_time = time.time()
            edge_mask = explain_function(model, h0[i], edge_index_set[batch_idx][i], targets[i], device, args)
            end_time = time.time()
            duration_seconds = end_time - start_time
            edge_masks.append(edge_mask)
            Time.append(duration_seconds)

        edge_masks_set.append(edge_masks)
    return (edge_masks_set, Time)
