import json
import os
import random
from dataset.gen_mutag import build_mutag
from dataset.gen_syn import build_syndata
from evaluate.accuracy import eval_accuracy
from evaluate.fidelity import eval_fidelity, eval_related_pred_gc, eval_related_pred_nc
from evaluate.mask_utils import clean_masks, get_size, get_sparsity, normalize_all_masks, transform_mask
from explainer.genmask import compute_edge_masks_gc, compute_edge_masks_nc
from gnn.eval import gnn_scores_gc, gnn_scores_nc
from gnn.model import GcnEncoderGraph, GcnEncoderNode
from gnn.train import train_graph_classification, train_node_classification
from utils.gen_utils import get_test_graphs, get_test_nodes
from utils.io_utils import check_dir, create_data_filename, create_model_filename, load_ckpt, save_checkpoint
from utils.parser_utils import arg_parse, get_data_args

import numpy as np
import torch


def main_syn(args):
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Generate, Save, Load data ###
    check_dir(args.data_save_dir)
    data_filename = create_data_filename(args)

    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
    else:
        data = build_syndata(args)
        torch.save(data, data_filename)

    data = data.to(device)
    args = get_data_args(data, args)

    ### Create, Train, Save, Load GNN model ###
    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GcnEncoderNode(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args=args
        )

    else:
        model = GcnEncoderNode(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args=args
        )
        train_node_classification(model, data, device, args)
        model.eval()
        results_train, results_test = gnn_scores_nc(model, data)
        save_checkpoint(model_filename, model, args, results_train, results_test)

    ckpt = load_ckpt(model_filename)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))

    ### Explain ###
    list_test_nodes = get_test_nodes(data, model, args)
    edge_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)

    ### Mask transformation ###
    # Replace Nan by 0, infinite by 0 and all value > 10e2 by 10e2
    edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
    edge_masks = clean_masks(edge_masks)

    # Normalize edge_masks to have value from 0 to 1
    edge_masks = normalize_all_masks(edge_masks)

    infos = {
        "dataset": args.dataset,
        "explainer": args.explainer_name,
        "number_of_edges": data.edge_index.size(1),
        "mask_sparsity_init": get_sparsity(edge_masks),
        "non_zero_values_init": get_size(edge_masks),
        "sparsity": args.sparsity,
        "threshold": args.threshold,
        "topk": args.topk,
        "num_test": args.num_test,
        "groundtruth target": args.true_label,
        "time": float(format(np.mean(Time), ".4f")),
    }
    print("__infos:" + json.dumps(infos))

    ### Accuracy Top ###
    accuracy_top = eval_accuracy(data, edge_masks, list_test_nodes, args, top_acc=True)
    print("__accuracy_top:" + json.dumps(accuracy_top))

    ### Mask transformation ###
    edge_masks = transform_mask(edge_masks, args)

    ### Accuracy ###
    accuracy = eval_accuracy(data, edge_masks, list_test_nodes, args, top_acc=False)
    print("__accuracy:" + json.dumps(accuracy))

    ### Fidelity ###
    related_preds = eval_related_pred_nc(model, data, edge_masks, list_test_nodes, device)
    fidelity = eval_fidelity(related_preds)
    print("__fidelity:" + json.dumps(fidelity))

    return


def main_mutag(args):
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Generate, Save, Load data ###
    check_dir(args.data_save_dir)
    data_filename = create_data_filename(args)

    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
    else:
        data = build_mutag(args)
        torch.save(data, data_filename)

    data = data.to(device)
    args = get_data_args(data, args)

    ### Create, Train, Save, Load GNN model ###
    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args=args
        )

    else:
        model = GcnEncoderGraph(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args=args
        )
        train_graph_classification(model, data, device, args)
        model.eval()
        results_train, results_test = gnn_scores_gc(model, data)
        save_checkpoint(model, args, results_train, results_test)

    ckpt = load_ckpt(model_filename)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))

    ### Explain ###
    list_test_graphs = get_test_graphs(data, model, args)
    edge_masks, Time = compute_edge_masks_gc(model, data, list_test_graphs, args)

    ### Mask transformation ###
    # Replace Nan by 0, infinite by 0 and all value > 10e2 by 10e2
    # edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
    # edge_masks = clean_masks(edge_masks)

    # Normalize edge_masks to have value from 0 to 1
    # edge_masks = normalize_all_masks(edge_masks)

    infos = {
        "dataset": args.dataset,
        "explainer": args.explainer_name,
        "number_of_edges": data.edge_index.size(1),
        "mask_sparsity_init": get_sparsity(edge_masks),
        "non_zero_values_init": get_size(edge_masks),
        "sparsity": args.sparsity,
        "threshold": args.threshold,
        "ktop": args.ktop,
        "num_test": args.num_test,
        "groundtruth target": args.true_label,
        "time": float(format(np.mean(Time), ".4f")),
    }
    print("__infos:" + json.dumps(infos))

    ### Mask transformation ###
    # edge_masks = transform_mask(edge_masks, args)

    ### Fidelity ###
    related_preds = eval_related_pred_gc(model, data, edge_masks, list_test_graphs, device)
    fidelity = eval_fidelity(related_preds)
    print("__fidelity:" + json.dumps(fidelity))

    return


if __name__ == "__main__":
    args = arg_parse()
    if args.explain_graph:
        main_mutag(args)
    else:
        main_syn(args)
