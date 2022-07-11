import json
import time
import os

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
import torch.nn.functional as F

from dataset.gen_syn import build_syndata
from dataset.gen_real import load_data_real
from dataset.data_utils import get_split, split_data
from evaluate.accuracy import eval_accuracy
from evaluate.fidelity import eval_fidelity, eval_related_pred_nc
from evaluate.mask_utils import clean_masks, get_mask_info, get_ratio_connected_components, get_size, get_sparsity, normalize_all_masks, transform_mask
from explainer.genmask import compute_edge_masks_nc
from gnn.eval import gnn_scores_nc, gnn_accuracy
from gnn.model import GCN, GcnEncoderNode
from gnn.train import train_real_nc, train_syn_nc
from utils.gen_utils import get_test_nodes, get_labels
from utils.io_utils import check_dir, create_data_filename, create_mask_filename, create_model_filename, load_ckpt, save_checkpoint
from utils.parser_utils import arg_parse, get_data_args, get_graph_size_args
from utils.plot_utils import plot_feat_importance, plot_masks_density

from new_method import new_method


def test_args():
    args = arg_parse()
    args.explainer_name = "new_method"
    args.num_test = 3
    return args


def test_compute_edge_masks_nc(list_test_nodes, model, data, device, args):
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
        edge_mask, node_feat_mask = new_method(
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




REAL_DATA = {"facebook": "FacebookPagePage", "cora": "Planetoid", "citeseer": "Planetoid", "pubmed": "Planetoid",
                "chameleon": "WikipediaNetwork", "squirrel": "WikipediaNetwork", 
                "ppi": "PPI", "actor": "Actor", 
                "texas": "WebKB", "cornell": "WebKB", "wisconsin": "WebKB"}
PLANETOIDS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
WEBKB = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}


def test_real(args):
    args.dataset = "cora"

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
    model = GCN(
            num_node_features=data.x.shape[1],
            hidden_dim=args.hidden_dim,
            num_classes=args.num_classes,
            dropout=args.dropout,
            num_layers=args.num_gc_layers,
            device=device,
    )
    ckpt = load_ckpt(model_filename, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))


    ### Explainer ###
    list_test_nodes = get_test_nodes(data, model, args)
    edge_masks, node_feat_masks, Time = test_compute_edge_masks_nc(list_test_nodes, model, data, device, args)
        

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

    print("Job successful!")
    return



def test_syn(args):

    args.dataset = "ba_house"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Generate, Save, Load data ###
    check_dir(args.data_save_dir)
    args = get_graph_size_args(args)
    data_filename = create_data_filename(args)
    data = torch.load(data_filename).to(device)
    args = get_data_args(data, args)
    print("_data_info: ", data.num_nodes, data.num_edges, args.num_classes)
    print("_val_data, test_data: ", data.val_mask.sum().item(), data.test_mask.sum().item())
    ### Create, Train, Save, Load GNN model ###
    model_filename = create_model_filename(args)
    model = GcnEncoderNode(
            args.input_dim,
            args.hidden_dim,
            args.output_dim,
            args.num_classes,
            args.num_gc_layers,
            args=args,
            device=device,
        )
    ckpt = load_ckpt(model_filename, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print("__gnn_train_scores: " + json.dumps(ckpt["results_train"]))
    print("__gnn_test_scores: " + json.dumps(ckpt["results_test"]))
    
    ### Explain ###
    list_test_nodes = get_test_nodes(data, model, args)
    edge_masks, node_feat_masks, Time = test_compute_edge_masks_nc(list_test_nodes, model, data, device, args)
        
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
        
    print("__infos:" + json.dumps(infos))

    
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
    print("Job successful!")
    return


if __name__ == "__main__":  
    args = test_args()

    print("Testing your method on synthetic data...")
    args.num_gc_layers, args.hidden_dim, args.output_dim, args.num_epochs, args.lr, args.weight_decay, args.dropout = 3, 20, 20, 1000, 0.001, 5e-3, 0.0
    test_syn(args)

    print("Testing your method on real data...")
    args.num_gc_layers, args.hidden_dim, args.output_dim,  args.num_epochs, args.lr, args.weight_decay, args.dropout = 2, 16, 16, 200, 0.01, 5e-4, 0.5
    test_real(args)

    print("Your method can be integrated to GraphFramEx :)")

    