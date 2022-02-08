import time

import torch
from torch.autograd import Variable
from utils.graph_utils import get_edge_index_set

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
        edge_mask = explain_function(model, node_idx, x, edge_index, targets[node_idx], device, args)
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


"""def compute_edge_masks_gc(model, test_data, device, args):
    explain_function = eval("explain_" + args.explainer_name)
    edge_index = get_edge_index_set(test_data)
    edge_masks = []
    Time = []

    for i in range(len(test_data)):
        data = test_data[i]
        h0 = Variable(torch.Tensor(data["feats"])).to(device)
        edges = Variable(torch.LongTensor(edge_index[i])).to(device)
        target = data["label"]
        start_time = time.time()
        edge_mask = explain_function(model, -1, h0, edges, target, device, args)
        end_time = time.time()
        duration_seconds = end_time - start_time
        Time.append(duration_seconds)
        edge_masks.append(edge_mask)
    return (edge_masks, Time)"""
