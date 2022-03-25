import json
import os
import random

import numpy as np
import torch
from sklearn import metrics
from torch_geometric.datasets import AmazonProducts, FacebookPagePage, Flickr, Planetoid, Reddit
from torch_geometric.loader import ClusterData, ClusterLoader
import torch.nn.functional as F

from dataset.gen_mutag import build_mutag
from dataset.gen_syn import build_syndata
from dataset.gen_planetoids import load_data_real
from dataset.data_utils import split_data
from dataset.mutag_utils import data_to_graph
from evaluate.accuracy import eval_accuracy
from evaluate.fidelity import eval_fidelity, eval_related_pred_gc, eval_related_pred_gc_batch, eval_related_pred_nc
from evaluate.mask_utils import clean_masks, get_mask_info, get_size, get_sparsity, normalize_all_masks, transform_mask
from explainer.genmask import compute_edge_masks_gc, compute_edge_masks_gc_batch, compute_edge_masks_nc
from gnn.eval import gnn_scores_gc, gnn_scores_nc, gnn_accuracy
from gnn.model import GCN, GcnEncoderGraph, GcnEncoderNode
from gnn.train import train_graph_classification, train_node_classification, train_real
from utils.gen_utils import gen_dataloader, get_test_graphs, get_test_nodes
from utils.graph_utils import get_edge_index_batch, split_batch
from utils.io_utils import check_dir, create_data_filename, create_model_filename, load_ckpt, save_checkpoint
from utils.parser_utils import arg_parse, get_data_args, get_graph_size_args
from utils.plot_utils import plot_expl_gc, plot_mask_density, plot_masks_density

REAL_DATA = {"reddit": "Reddit", "facebook": "FacebookPagePage", "flickr": "Flickr", "amazon": "AmazonProducts", "cora": "Planetoid", "citeseer": "Planetoid", "pubmed": "Planetoid"}
PLANETOIDS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def main_real(args):

    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_dir(args.data_save_dir)
    data_dir = os.path.join(args.data_save_dir, args.dataset)
    check_dir(data_dir)
    data_filename = f"{data_dir}/processed/data.pt"
    if not os.path.isfile(data_filename):
        if REAL_DATA[args.dataset] == "Planetoid":
            Planetoid(args.data_save_dir, name=PLANETOIDS[args.dataset])
            origin_dir = os.path.join(args.data_save_dir, PLANETOIDS[args.dataset])
            os.rename(origin_dir, data_dir)
        else:
            eval(REAL_DATA[args.dataset])(data_dir)

    data = load_data_real(data_filename)
    if args.dataset == "facebook":
        data = split_data(data, args)
    data = data.to(device)

    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GCN(
            num_node_features=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=data.y.max().item() + 1,
            dropout=args.dropout,
            num_layers=args.num_gc_layers,
            device=device,
        )
    else:
        model = GCN(
            num_node_features=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=data.y.max().item() + 1,
            dropout=args.dropout,
            num_layers=args.num_gc_layers,
            device=device,
        )
        train_real(model, data, device, args)
        results_train, results_test = gnn_scores_nc(model, data, args, device)
        save_checkpoint(model_filename, model, args, results_train, results_test)

    ckpt = load_ckpt(model_filename, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))

    ### Explainer ###
    list_test_nodes = get_test_nodes(data, model, args)
    edge_masks, node_feat_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)
    
    ### Mask normalisation and cleaning ###
    edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
    edge_masks = clean_masks(edge_masks)
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
        "groundtruth target": args.true_label_as_target,
        "time": float(format(np.mean(Time), ".4f")),
    }
    print("__infos:" + json.dumps(infos))

    ### Mask transformation ###
    edge_masks = transform_mask(edge_masks, args)
    if (eval(args.hard_mask)==False)&(args.seed==0):
        plot_masks_density(edge_masks, args)
    print("__mask_info:" + json.dumps(get_mask_info(edge_masks)))

    ### Fidelity ###
    related_preds = eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_test_nodes, device, args)
    fidelity = eval_fidelity(related_preds, args)
    print("__fidelity:" + json.dumps(fidelity))









def main_syn(args):
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Generate, Save, Load data ###
    check_dir(args.data_save_dir)
    args = get_graph_size_args(args)
    data_filename = create_data_filename(args)
    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
    else:

        data = build_syndata(args)
        torch.save(data, data_filename)

    data = data.to(device)
    args = get_data_args(data, args)
    print("_data_info: ", data.num_nodes, data.num_edges, args.num_classes)
    print("_val_data, test_data: ", data.val_mask.sum().item(), data.test_mask.sum().item())
    ### Create, Train, Save, Load GNN model ###
    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args=args,
            device=device,
        )

    else:
        model = GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args=args,
            device=device,
        )
        train_node_classification(model, data, device, args)
        model.eval()
        output = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        acc_test = gnn_accuracy(output[data.test_mask], data.y[data.test_mask])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

        results_train, results_test = gnn_scores_nc(model, data, args, device)
        save_checkpoint(model_filename, model, args, results_train, results_test)

    ckpt = load_ckpt(model_filename, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))
    
    ### Explain ###
    list_test_nodes = get_test_nodes(data, model, args)
    edge_masks, node_feat_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)
    # plot_mask_density(edge_masks, args)

    ### Mask normalisation and cleaning ###
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
        "groundtruth target": args.true_label_as_target,
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
    related_preds = eval_related_pred_nc(model, data, edge_masks, list_test_nodes, device, args)
    fidelity = eval_fidelity(related_preds, args)
    print("__fidelity:" + json.dumps(fidelity))

    return


def main_mutag(args, batch=True):
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

    ### Crucial step: converting Pytorch Data object to networkx Graph object with features: adj, feat, ...
    args = get_data_args(data, args)
    ### Create, Train, Save, Load GNN model ###
    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GcnEncoderGraph(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args=args,
            device=device,
        )

    else:
        model = GcnEncoderGraph(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args=args,
            device=device,
        )
        graphs = data_to_graph(data)
        train_graph_classification(model, graphs, device, args)
        model.eval()
        results_train, results_test = gnn_scores_gc(model, graphs, args, device)
        save_checkpoint(model_filename, model, args, results_train, results_test)

    ckpt = load_ckpt(model_filename, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))

    ### Explain ###

    test_data = get_test_graphs(data, args)
    test_Data_pytorch = test_data

    if batch:
        test_data = data_to_graph(test_data)
        test_data = gen_dataloader(test_data, args)
        edge_masks_set, Time = compute_edge_masks_gc_batch(model, test_data, device, args)
        ### Mask transformation ###
        # Replace Nan by 0, infinite by 0 and all value > 10e2 by 10e2
        edge_masks = np.hstack(edge_masks_set)
        edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
        edge_masks = clean_masks(edge_masks)

        # Normalize edge_masks to have value from 0 to 1
        edge_masks = normalize_all_masks(edge_masks)

        infos = {
            "dataset": args.dataset,
            "explainer": args.explainer_name,
            "number_of_edges": np.mean([len(edge_mask) for edge_mask in edge_masks]),
            "mask_sparsity_init": get_sparsity(edge_masks),
            "non_zero_values_init": get_size(edge_masks),
            "sparsity": args.sparsity,
            "threshold": args.threshold,
            "topk": args.topk,
            "num_test": args.num_test,
            "groundtruth target": args.true_label_as_target,
            "time": float(format(np.mean(Time), ".4f")),
        }
        print("__infos:" + json.dumps(infos))

        ### Mask transformation ###
        edge_masks = transform_mask(edge_masks, args)
        edge_masks_set = split_batch(edge_masks, args.batch_size)

        ### Fidelity ###
        edge_index_set = get_edge_index_batch(test_data)
        related_preds = eval_related_pred_gc_batch(model, test_data, edge_index_set, edge_masks_set, device, args)
        fidelity = eval_fidelity(related_preds)
        print("__fidelity:" + json.dumps(fidelity))

    else:
        edge_masks, Time = compute_edge_masks_gc(model, test_data, device, args)
        ### Mask transformation ###
        # Replace Nan by 0, infinite by 0 and all value > 10e2 by 10e2
        edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
        edge_masks = clean_masks(edge_masks)

        # Normalize edge_masks to have value from 0 to 1
        edge_masks = normalize_all_masks(edge_masks)

        infos = {
            "dataset": args.dataset,
            "explainer": args.explainer_name,
            "number_of_edges": np.mean([len(edge_mask) for edge_mask in edge_masks]),
            "mask_sparsity_init": get_sparsity(edge_masks),
            "non_zero_values_init": get_size(edge_masks),
            "sparsity": args.sparsity,
            "threshold": args.threshold,
            "topk": args.topk,
            "num_test": args.num_test,
            "groundtruth target": args.true_label_as_target,
            "time": float(format(np.mean(Time), ".4f")),
        }
        print("__infos:" + json.dumps(infos))

        ### Mask transformation ###
        edge_masks = transform_mask(edge_masks, args)

        ### Fidelity ###
        related_preds = eval_related_pred_gc(model, test_data, edge_masks, device, args)
        fidelity = eval_fidelity(related_preds)
        print("__fidelity:" + json.dumps(fidelity))

    if eval(args.draw_graph):
        plot_expl_gc(test_Data_pytorch, edge_masks, args)

    return


if __name__ == "__main__":
    args = arg_parse()
    if eval(args.explain_graph):
        main_mutag(args)
    elif args.dataset.startswith("syn"):
        args.num_gc_layers, args.hidden_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 3, 20, 1000, 0.001, 5e-3, 0.0
        main_syn(args)
    elif args.dataset in PLANETOIDS.keys():
        args.num_gc_layers, args.hidden_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 16, 200, 0.01, 5e-4, 0.5
        main_real(args)
    elif args.dataset == "facebook":
        args.num_gc_layers, args.hidden_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 16, 200, 0.01, 5e-4, 0.5
        main_real(args)
    elif args.dataset == "flickr":
        args.num_gc_layers, args.hidden_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 20, 300, 0.005, 0, 0
        main_real(args)
    elif args.dataset == "ebay":
        args.num_gc_layers, args.hidden_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 20, 300, 0.005, 0, 0
        main_real(args)
    else:
        pass
