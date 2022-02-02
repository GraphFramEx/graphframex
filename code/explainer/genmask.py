import time
import torch
from torch.autograd import Variable
from dataset.mutag_utils import data_to_graph, gen_dataloader

from utils.graph_utils import get_edge_index_set
from explainer.method import *


def compute_edge_masks_nc(list_test_nodes, model, data, device, args):
    explain_function = eval("explain_" + args.explainer_name)
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


def compute_edge_masks_gc(model, dataset, device, args):
    graphs = data_to_graph(dataset)
    dataset = gen_dataloader(graphs, args)
    explain_function = eval("explain_" + args.explainer_name)
    edge_index_set = get_edge_index_set(dataset)
    edge_masks_set = []
    Time = []

    for batch_idx, data in enumerate(dataset):
        edge_masks = []

        h0 = Variable(data["feats"].float()).to(device)
        targets = data["label"].long().numpy()

        for i in range(len(edge_index_set[batch_idx])):
            start_time = time.time()
            edge_mask = explain_function(model, -1, h0[i], edge_index_set[batch_idx][i], targets[i], device)
            end_time = time.time()
            duration_seconds = end_time - start_time
            edge_masks.append(edge_mask)
            Time.append(duration_seconds)

        edge_masks_set.append(edge_masks)
    return (edge_masks_set, Time)
