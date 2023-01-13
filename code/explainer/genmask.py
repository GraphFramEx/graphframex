import pickle
import time
from utils.io_utils import create_mask_filename
from utils.gen_utils import get_labels

import torch

from explainer.graph_explainer import *
from explainer.node_explainer import *


def compute_edge_masks_nc(list_test_nodes, model, data, device, args):
    explain_function = eval("explain_" + args.explainer_name + "_node")
    Time = []
    edge_masks, node_feat_masks = [], []
    if eval(args.true_label_as_target):
        targets = data.y
    else:
        out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        targets = torch.LongTensor(get_labels(out.detach().cpu().numpy())).to(device)
    t0 = time.time()
    for node_idx in list_test_nodes:
        start_time = time.time()
        edge_mask, node_feat_mask = explain_function(
            model, data, node_idx, targets[node_idx], device, args
        )
        end_time = time.time()
        duration_seconds = end_time - start_time
        Time.append(duration_seconds)
        edge_masks.append(edge_mask)
        node_feat_masks.append(node_feat_mask)
        t1 = time.time()
        if t1 - t0 > args.time_limit:
            print("Time limit reached")
            break
    args.num_test_final = len(edge_masks)
    return edge_masks, node_feat_masks, Time



def compute_masks(list_test_nodes, model, data, device, args):
        mask_filename = create_mask_filename(args)
        if (os.path.isfile(mask_filename)) & (args.explainer_name not in ["sa", "ig"]):
            with open(mask_filename, 'rb') as f:
                w_list = pickle.load(f)
            list_test_nodes, edge_masks, node_feat_masks, Time = tuple(w_list)
        else:
            edge_masks, node_feat_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)
            if args.explainer_name not in ["sa", "ig"]:
                with open(mask_filename, 'wb') as f:
                    pickle.dump([list_test_nodes, edge_masks, node_feat_masks, Time], f)
        return list_test_nodes, edge_masks, node_feat_masks, Time