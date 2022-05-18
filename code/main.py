from hashlib import new
import json
import os
import random
import shutil
import pickle

import numpy as np
import torch
from sklearn import metrics
from torch_geometric.datasets import FacebookPagePage, Planetoid, WikipediaNetwork, PPI, Actor, WebKB
from torch_geometric.loader import ClusterData, ClusterLoader
import torch.nn.functional as F
from scipy.special import softmax

from dataset.gen_mutag import build_mutag
from dataset.gen_syn import build_syndata
from dataset.gen_real import load_data_real
from dataset.data_utils import get_split, split_data
from dataset.mutag_utils import data_to_graph
from evaluate.accuracy import eval_accuracy
from evaluate.fidelity import eval_fidelity, eval_related_pred_gc, eval_related_pred_gc_batch, eval_related_pred_nc
from evaluate.mask_utils import clean_masks, get_mask_info, get_ratio_connected_components, get_size, get_sparsity, mask_to_shape, normalize_all_masks, transform_mask
from explainer.genmask import compute_edge_masks_gc, compute_edge_masks_gc_batch, compute_edge_masks_nc
from gnn.eval import gnn_scores_gc, gnn_scores_nc, gnn_accuracy
from gnn.model import GCN, GcnEncoderGraph, GcnEncoderNode
from gnn.train import train_graph_classification, train_node_classification, train_real
from utils.gen_utils import gen_dataloader, get_labels, get_test_graphs, get_test_nodes
from utils.graph_utils import get_edge_index_batch, split_batch
from utils.io_utils import check_dir, create_data_filename, create_mask_filename, create_model_filename, load_ckpt, save_checkpoint
from utils.parser_utils import arg_parse, get_data_args, get_graph_size_args
from utils.plot_utils import plot_expl_gc, plot_feat_importance, plot_mask_density, plot_masks_density

REAL_DATA = {"facebook": "FacebookPagePage", "cora": "Planetoid", "citeseer": "Planetoid", "pubmed": "Planetoid",
                "chameleon": "WikipediaNetwork", "squirrel": "WikipediaNetwork", 
                "ppi": "PPI", "actor": "Actor", 
                "texas": "WebKB", "cornell": "WebKB", "wisconsin": "WebKB"}
PLANETOIDS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
WEBKB = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}


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
        elif REAL_DATA[args.dataset] == "WebKB":
            WebKB(args.data_save_dir, name=WEBKB[args.dataset])
        elif REAL_DATA[args.dataset] == "WikipediaNetwork":
            WikipediaNetwork(args.data_save_dir, name=args.dataset)
            origin_dir = os.path.join(data_dir, "geom_gcn")
            # fetch all files
            for folder_name in os.listdir(origin_dir):
                # construct full file path
                source =  os.path.join(origin_dir, folder_name)
                destination = os.path.join(data_dir, folder_name)
                # move only folder
                print(f"Moving {source} to {destination}")
                if os.path.isdir(source):
                    print('moving folder {} to {}'.format(source, destination))
                    shutil.move(source, destination)
            shutil.rmtree(origin_dir, ignore_errors=True)
        else:
            eval(REAL_DATA[args.dataset])(data_dir)

    data = load_data_real(data_filename)
    if args.dataset == "facebook":
        data = split_data(data, args)
    if data.train_mask.dim() > 1:
        data = get_split(data, args)
    data = data.to(device)

    args.num_classes = data.y.max().item() + 1

    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GCN(
            num_node_features=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            dropout=args.dropout,
            num_layers=args.num_gc_layers,
            device=device,
        )
    else:
        model = GCN(
            num_node_features=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
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
    mask_filename = create_mask_filename(args)

    if args.dataset.startswith("ebay"):
        edge_masks, node_feat_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)
        
    else:
        if (os.path.isfile(mask_filename)) & (args.explainer_name not in ["sa", "ig"]):
            with open(mask_filename, 'rb') as f:
                w_list = pickle.load(f)
            edge_masks, node_feat_masks, Time = tuple(w_list)
        else:
            edge_masks, node_feat_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)
            if args.explainer_name not in ["sa", "ig"]:
                with open(mask_filename, 'wb') as f:
                    pickle.dump([edge_masks, node_feat_masks, Time], f)

    args.E = False if edge_masks[0] is None else True
    args.NF = False if node_feat_masks[0] is None else True
    if args.NF:
        if node_feat_masks[0].size<=1:
            args.NF = False
            print("No node feature mask")
    args.num_test_final = len(edge_masks) if args.E else None

    infos = {
            "dataset": args.dataset,
            "explainer": args.explainer_name,
            "number_of_edges": data.edge_index.size(1),
            "num_test": args.num_test,
            "num_test_final": args.num_test_final,
            "groundtruth target": args.true_label_as_target,
            "time": float(format(np.mean(Time), ".4f")),}

    
    if args.E:
        ### Mask normalisation and cleaning ###
        edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
        edge_masks = clean_masks(edge_masks)
        print("__initial_edge_mask_infos:" + json.dumps(get_mask_info(edge_masks, data.edge_index)))

        infos["edge_mask_sparsity_init"] = get_sparsity(edge_masks)
        infos["edge_mask_size_init"] = get_size(edge_masks)
        infos["edge_mask_connected_init"] = get_ratio_connected_components(edge_masks, data.edge_index)
        
    if args.NF:
        ### Mask normalisation and cleaning ###
        node_feat_masks = [node_feat_mask.astype("float") for node_feat_mask in node_feat_masks]
        node_feat_masks = clean_masks(node_feat_masks)
        
        #infos["node_feat_mask_sparsity_init"] = get_sparsity(node_feat_masks)
        #infos["node_feat_mask_size_init"] = get_size(node_feat_masks)
        
        if (eval(args.hard_mask)==False)&(args.seed==10):
            plot_masks_density(node_feat_masks, args, type="node_feat")
            plot_feat_importance(node_feat_masks, args)

    print("__infos:" + json.dumps(infos))


    if (not args.strategy)|(not args.params_list):
        print("Masks are not transformed")

        ### Fidelity ###
        related_preds = eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_test_nodes, device, args)
        fidelity = eval_fidelity(related_preds, args)
        fidelity_scores = {key: value for key, value in sorted(fidelity.items() | params_transf.items())}
        print("__fidelity:" + json.dumps(fidelity_scores))


    else: 
        print("Masks are transformed with strategy: " + args.strategy)
        params_lst = [eval(i) for i in args.params_list.split(',')]
    
        edge_masks_ori = edge_masks.copy()
        for param in params_lst:
            params_transf = {args.strategy: param}

            ### Mask transformation ###
            edge_masks = transform_mask(edge_masks_ori, data, param, args)
            if (eval(args.hard_mask)==False)&(args.seed==10):
                plot_masks_density(edge_masks, args, type="edge")
            transformed_mask_infos = {key: value for key, value in sorted(get_mask_info(edge_masks, data.edge_index).items() | params_transf.items())}
            print("__transformed_mask_infos:" + json.dumps(transformed_mask_infos))

            ### Fidelity ###
            related_preds = eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_test_nodes, device, args)
            fidelity = eval_fidelity(related_preds, args)
            fidelity_scores = {key: value for key, value in sorted(fidelity.items() | params_transf.items())}
            print("__fidelity:" + json.dumps(fidelity_scores))









def main_syn(args):

    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
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
    mask_filename = create_mask_filename(args)
    if os.path.isfile(mask_filename):
        with open(mask_filename, 'rb') as f:
            w_list = pickle.load(f)
        edge_masks, node_feat_masks, Time = tuple(w_list)
    else:
        edge_masks, node_feat_masks, Time = compute_edge_masks_nc(list_test_nodes, model, data, device, args)
        with open(mask_filename, 'wb') as f:
            pickle.dump([edge_masks, node_feat_masks, Time], f)

    args.E = False if edge_masks[0] is None else True
    args.NF = False if node_feat_masks[0] is None else True
    if args.NF:
        if node_feat_masks[0].size<=1:
            args.NF = False
            print("No node feature mask")
    args.num_test_final = len(edge_masks) if args.E else None


    infos = {
            "dataset": args.dataset,
            "explainer": args.explainer_name,
            "number_of_edges": data.edge_index.size(1),
            "num_test": args.num_test,
            "num_test_final": args.num_test_final,
            "groundtruth target": args.true_label_as_target,
            "time": float(format(np.mean(Time), ".4f")),}
    
    if args.E:
        ### Mask normalisation and cleaning ###
        edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
        edge_masks = clean_masks(edge_masks)
        print("__initial_edge_mask_infos:" + json.dumps(get_mask_info(edge_masks, data.edge_index)))

        infos["edge_mask_sparsity_init"] = get_sparsity(edge_masks)
        infos["edge_mask_size_init"] = get_size(edge_masks)
        infos["edge_mask_connected_init"] = get_ratio_connected_components(edge_masks, data.edge_index)
        
        
    if args.NF:
        ### Mask normalisation and cleaning ###
        node_feat_masks = [node_feat_mask.astype("float") for node_feat_mask in node_feat_masks]
        node_feat_masks = clean_masks(node_feat_masks)
        
        infos["node_feat_mask_sparsity_init"] = get_sparsity(node_feat_masks)
        infos["node_feat_mask_size_init"] = get_size(node_feat_masks)
        
        if (eval(args.hard_mask)==False)&(args.seed==10):
            plot_masks_density(node_feat_masks, args, type="node_feat")
            plot_feat_importance(node_feat_masks, args)

    print("__infos:" + json.dumps(infos))

    if eval(args.top_acc):
        ### Accuracy Top ###
        accuracy_top = eval_accuracy(data, edge_masks, list_test_nodes, args, top_acc=True)
        print("__accuracy_top:" + json.dumps(accuracy_top))
    
    else:

        if (args.strategy not in ["topk", "sparsity", "threshold"])|(eval(args.params_list) is None):
            print("Masks are not transformed")
            args.param = None
            params_transf = {"strategy": args.strategy, "params_list": args.params_list}

            ### Accuracy ###
            accuracy = eval_accuracy(data, edge_masks, list_test_nodes, args, top_acc=False)
            accuracy_scores = {key: value for key, value in sorted(accuracy.items() | params_transf.items())}
            print("__accuracy:" + json.dumps(accuracy_scores))

            ### Fidelity ###
            related_preds = eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_test_nodes, device, args)
            fidelity = eval_fidelity(related_preds, args)
            fidelity_scores = {key: value for key, value in sorted(fidelity.items() | params_transf.items())}
            print("__fidelity:" + json.dumps(fidelity_scores))


        else: 
            print("Masks are transformed with strategy: " + args.strategy)
            params_lst = [eval(i) for i in args.params_list.split(',')]
        
            edge_masks_ori = edge_masks.copy()
            for param in params_lst:
                params_transf = {args.strategy: param}
                args.param = param

                ### Mask transformation ###
                edge_masks = transform_mask(edge_masks_ori, data, param, args)
                if (eval(args.hard_mask)==False)&(args.seed==10):
                    plot_masks_density(edge_masks, args, type="edge")
                transformed_mask_infos = {key: value for key, value in sorted(get_mask_info(edge_masks, data.edge_index).items() | params_transf.items())}
                print("__transformed_mask_infos:" + json.dumps(transformed_mask_infos))

                ### Accuracy ###
                accuracy = eval_accuracy(data, edge_masks, list_test_nodes, args, top_acc=False)
                accuracy_scores = {key: value for key, value in sorted(accuracy.items() | params_transf.items())}
                print("__accuracy:" + json.dumps(accuracy_scores))

                ### Fidelity ###
                related_preds = eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_test_nodes, device, args)
                fidelity = eval_fidelity(related_preds, args)
                fidelity_scores = {key: value for key, value in sorted(fidelity.items() | params_transf.items())}
                print("__fidelity:" + json.dumps(fidelity_scores))

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
        args.num_gc_layers, args.hidden_dim, args.output_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 3, 20, 20, 1000, 0.001, 5e-3, 0.0
        main_syn(args)
    elif args.dataset in REAL_DATA.keys():
        if args.dataset in WEBKB.keys():
            args.num_gc_layers, args.hidden_dim, args.output_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 32, 32, 400, 0.001, 5e-3, 0.2
        else: 
            args.num_gc_layers, args.hidden_dim, args.output_dim,  args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 16, 16, 200, 0.01, 5e-4, 0.5
        main_real(args)
    elif args.dataset.startswith("ebay"):
        args.num_gc_layers, args.hidden_dim, args.output_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 32, 32, 500, 0.001, 5e-4, 0.5
        main_real(args)
    else:
        pass
