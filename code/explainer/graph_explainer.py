import os
import numpy as np
import random
import torch
import torch.nn.functional as F
import random
import time
import json
import dill
import argparse
from copy import deepcopy
from captum.attr import IntegratedGradients, Saliency
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_dense_adj
from code.explainer.gnnlrp import GNN_LRP
from code.explainer.pgexplainer import PGExplainer
from code.utils.math_utils import sigmoid
from utils.gen_utils import (
    filter_existing_edges,
    get_cmn_edges,
    from_edge_index_to_adj_torch,
    from_adj_to_edge_index_torch,
    from_edge_index_to_adj,
    from_adj_to_edge_index,
    get_neighbourhood,
    normalize_adj,
    sample_large_graph,
)
from utils.io_utils import write_to_json
from gnn.model import GCNConv, GATConv, GINEConv, TransformerConv

from explainer.gnnexplainer import TargetedGNNExplainer
from explainer.pgmexplainer import Graph_Explainer
from explainer.subgraphx import SubgraphX
from explainer.gradcam import GraphLayerGradCam
from explainer.cfgnnexplainer import CFExplainer
from explainer.graphcfe import GraphCFE, train, test, baseline_cf, add_list_in_dict, compute_counterfactual
from explainer.gflowexplainer import GFlowExplainer, gflow_parse_args
from explainer.diffexplainer import DiffExplainer, diff_parse_args
from explainer.rcexplainer import RCExplainer_Batch, train_rcexplainer
from explainer.explainer_utils.rcexplainer.rc_train import test_policy
from gendata import get_dataset, get_dataloader
from explainer.gsat import GSAT, ExtractorMLP, gsat_get_config
from explainer.explainer_utils.gsat import Writer, init_metric_dict, save_checkpoint, load_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau



def get_all_convolution_layers(model):
    layers = []
    for module in model.modules():
        if (
            isinstance(module, GCNConv)
            or isinstance(module, GATConv)
            or isinstance(module, GINEConv)
            or isinstance(module, TransformerConv)
        ):
            layers.append(module)
    return layers


def gpu_to_cpu(data, device):
    data.x = torch.FloatTensor(data.x.cpu().numpy().copy()).to(device)
    data.edge_index = torch.LongTensor(data.edge_index.cpu().numpy().copy()).to(device)
    data.edge_attr = torch.FloatTensor(data.edge_attr.cpu().numpy().copy()).to(device)
    return data


def model_forward_graph(x, model, edge_index, edge_attr):
    out = model(x, edge_index, edge_attr)
    return out


def node_attr_to_edge(edge_index, node_mask):
    edge_mask = np.zeros(edge_index.shape[1])
    edge_mask += node_mask[edge_index[0].cpu().numpy()]
    edge_mask += node_mask[edge_index[1].cpu().numpy()]
    return edge_mask


#### Baselines ####
def explain_random_graph(model, data, target, device, **kwargs):
    edge_mask = np.random.uniform(size=data.edge_index.shape[1])
    node_feat_mask = np.random.uniform(size=data.x.shape[1])
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_sa_graph(model, data, target, device, **kwargs):
    saliency = Saliency(model_forward_graph)
    input_mask = data.x.clone().requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(
        input_mask,
        target=target,
        additional_forward_args=(model, data.edge_index, data.edge_attr),
        abs=False,
    )
    # 1 node feature mask per node.
    node_feat_mask = saliency_mask.cpu().numpy()
    node_attr = node_feat_mask.sum(axis=1)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_ig_graph(model, data, target, device, **kwargs):
    ig = IntegratedGradients(model_forward_graph)
    input_mask = data.x.clone().requires_grad_(True).to(device)
    ig_mask = ig.attribute(
        input_mask,
        target=target,
        additional_forward_args=(model, data.edge_index, data.edge_attr),
        internal_batch_size=input_mask.shape[0],
    )
    node_feat_mask = ig_mask.cpu().detach().numpy()
    node_attr = node_feat_mask.sum(axis=1)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_occlusion_graph(model, data, target, device, **kwargs):
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    if target is None:
        pred_probs = model(data)[0].cpu().detach().numpy()
        pred_prob = pred_probs.max()
        target = pred_probs.argmax()
    else:
        pred_prob = 1
    g = to_networkx(data)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        # if include_edges is not None and not include_edges[i].item():
        # continue
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in g.edges():
            edge_occlusion_mask[i] = False
            prob = model(
                data.x,
                data.edge_index[:, edge_occlusion_mask],
                data.edge_attr[edge_occlusion_mask],
            )[0][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask.astype("float"), None


def explain_basic_gnnexplainer_graph(model, data, target, device, **kwargs):
    data = gpu_to_cpu(data, device)
    explainer = TargetedGNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        return_type="prob",
        epochs=1000,
        edge_ent=kwargs["edge_ent"],
        edge_size=kwargs["edge_size"],
        allow_edge_mask=True,
        allow_node_mask=False,
        device=device,
    )
    _, edge_mask = explainer.explain_graph_with_target(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=target,
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_gnnexplainer_graph(model, data, target, device, **kwargs):
    data = gpu_to_cpu(data, device)
    explainer = TargetedGNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        return_type="prob",
        epochs=1000,
        edge_ent=kwargs["edge_ent"],
        edge_size=kwargs["edge_size"],
        allow_edge_mask=True,
        allow_node_mask=True,
        device=device,
    )
    node_feat_mask, edge_mask = explainer.explain_graph_with_target(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=target,
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    node_feat_mask = node_feat_mask.cpu().detach().numpy()
    return edge_mask, node_feat_mask


def explain_pgmexplainer_graph(model, data, target, device, **kwargs):
    explainer = Graph_Explainer(
        model, data.edge_index, data.edge_attr, data.x, device=device, print_result=0
    )
    explanation = explainer.explain(
        num_samples=1000,
        percentage=10,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1,
    )
    node_attr = np.zeros(data.x.shape[0])
    for node, p_value in explanation.items():
        node_attr[node] = 1 - p_value
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), None


def explain_subgraphx_graph(model, data, target, device, **kwargs):
    subgraphx = SubgraphX(
        model,
        kwargs["num_classes"],
        device,
        num_hops=kwargs["num_layers"],
        explain_graph=True,
        rollout=20,
        min_atoms=4,
        expand_atoms=14,
        high2low=True,
        sample_num=50,
        reward_method="mc_shapley",
        subgraph_building_method="zero_filling",
        local_radius=4,
    )
    edge_mask = subgraphx.explain(
        data.x,
        data.edge_index,
        data.edge_attr,
        max_nodes=kwargs["num_top_edges"],
        label=target,
    )
    return edge_mask.astype("float"), None


def explain_gradcam_graph(model, data, target, device, **kwargs):
    # Captum default implementation of LayerGradCam does not average over nodes for different channels because of
    # different assumptions on tensor shapes
    input_mask = data.x.clone().requires_grad_(True).to(device)
    layers = get_all_convolution_layers(model)
    node_attrs = []
    for layer in layers:
        layer_gc = GraphLayerGradCam(model_forward_graph, layer)
        node_attr = layer_gc.attribute(
            input_mask,
            target=target,
            additional_forward_args=(
                model,
                data.edge_index,
                data.edge_attr,
            ),
        )
        node_attrs.append(node_attr.squeeze().cpu().detach().numpy())
    node_attr = np.array(node_attrs).mean(axis=0)
    edge_mask = sigmoid(node_attr_to_edge(data.edge_index, node_attr))
    return edge_mask.astype("float"), None


def explain_pgexplainer_graph(model, data, target, device, **kwargs):
    seed = kwargs['seed']
    pgexplainer = PGExplainer(
        model,
        in_channels=kwargs["hidden_dim"] * 2,
        device=device,
        num_hops=kwargs["num_layers"],
        explain_graph=True,
    )
    dataset_name = kwargs["dataset_name"]
    subdir = os.path.join(kwargs["model_save_dir"], "pgexplainer")
    os.makedirs(subdir, exist_ok=True)
    pgexplainer_saving_path = os.path.join(subdir, f"pgexplainer_{dataset_name}_{str(device)}_{seed}.pth")
    if os.path.isfile(pgexplainer_saving_path):
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    else:
        data = sample_large_graph(data)
        t0 = time.time()
        pgexplainer.train_explanation_network(kwargs["dataset"][:200])
        train_time = time.time() - t0
        print("Save PGExplainer model...")
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        train_time_file = os.path.join(subdir, f"pgexplainer_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
        
    embed = model.get_emb(data=data)
    _, edge_mask = pgexplainer.explain(
        data.x, data.edge_index, data.edge_attr, embed=embed, tmp=1.0, training=False
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_gnnlrp_graph(model, data, target, device, **kwargs):
    gnnlrp = GNN_LRP(model, explain_graph=True)
    walks, edge_mask = gnnlrp(
        data.x,
        data.edge_index,
        data.edge_attr,
        device,
        explain_graph=True,
        num_classes=kwargs["num_classes"],
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_cfgnnexplainer_graph(model, data, target, device, **kwargs):
    n_momentum, num_epochs, beta, optimizer, lr = 0.9, 500, 0, "SGD", 0.1
    model.eval()
    features, labels = data.x, data.y
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0]).squeeze(0)
    # adj = from_edge_index_to_adj(data.edge_index, data.edge_attr, data.x.shape[0])
    # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
    explainer = CFExplainer(
        model=model,
        cf_model_name = kwargs["model_name"],
        adj=adj,
        feat=features,
        edge_feat=data.edge_attr,
        n_hid=kwargs["hidden_dim"],
        dropout=kwargs["dropout"],
        readout=kwargs["readout"],
        edge_dim=kwargs["edge_dim"],
        num_layers=kwargs["num_layers"],
        labels=labels,
        y_pred_orig=target,
        num_classes=kwargs["num_classes"],
        beta=beta,
        device=device,
    )
    cf_example = explainer.explain(
        cf_optimizer=optimizer,
        lr=lr,
        n_momentum=n_momentum,
        num_epochs=num_epochs,
    )
    if cf_example == []:
        return None, None
    else:
        perturb_edges = cf_example[0][0]
        print("Perturbed edges: ", perturb_edges)
        edge_mask = filter_existing_edges(perturb_edges, data.edge_index)
        if edge_mask:
            print("Edge mask: ", edge_mask)
            return edge_mask, None
        else:
            return None, None


def explain_graphcfe_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    y_cf_all = kwargs['y_cf_all']
    seed = kwargs["seed"]

    # data loader
    train_size = min(len(kwargs["dataset"]), 500)
    explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
    explain_dataset = kwargs["dataset"][explain_dataset_idx]
    dataloader_params = {
        "batch_size": kwargs["batch_size"],
        "random_split_flag": kwargs["random_split_flag"],
        "data_split_ratio": kwargs["data_split_ratio"],
        "seed": kwargs["seed"],
    }
    loader, _, _, _ = get_dataloader(explain_dataset, **dataloader_params)
    
    # metrics
    metrics = ['validity', 'proximity_x', 'proximity_a']

    subdir = os.path.join(kwargs["model_save_dir"], "graphcfe")
    os.makedirs(subdir, exist_ok=True)
    graphcfe_saving_path = os.path.join(subdir, f"graphcfe_{dataset_name}_{str(device)}_{seed}.pth")
     # model
    init_params = {'hidden_dim': kwargs["hidden_dim"], 'dropout': kwargs["dropout"], 'num_node_features': kwargs["num_node_features"], 'max_num_nodes': kwargs["max_num_nodes"]}
    graphcfe_model = GraphCFE(init_params=init_params, device=device)

    if os.path.isfile(graphcfe_saving_path):
        print("Load saved GraphCFE model...")
        state_dict = torch.load(graphcfe_saving_path)
        graphcfe_model.load_state_dict(state_dict)
        graphcfe_model = graphcfe_model.to(device)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        graphcfe_model = graphcfe_model.to(device)
        train_params = {'epochs': 4000, 'model': graphcfe_model, 'pred_model': model, 'optimizer': optimizer,
                        'y_cf': y_cf_all,
                        'train_loader': loader['train'], 'val_loader': loader['eval'], 'test_loader': loader['test'],
                        'dataset': dataset_name, 'metrics': metrics, 'save_model': False}
        t0 = time.time()
        train(train_params)
        train_time = time.time() - t0
        print("Save GraphCFE model...")
        torch.save(graphcfe_model.state_dict(), graphcfe_saving_path)
        train_time_file = os.path.join(subdir, f"graphcfe_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
    # test
    test_params = {'model': graphcfe_model, 'dataset': dataset_name, 'data_loader': loader['test'], 'pred_model': model,
                       'metrics': metrics, 'y_cf': y_cf_all}
    eval_results = test(test_params)
    results_all_exp = {}
    for k in metrics:
        results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())
    for k in eval_results:
        if isinstance(eval_results[k], list):
            print(k, ": ", eval_results[k])
        else:
            print(k, f": {eval_results[k]:.4f}")

    # baseline
    # num_rounds, type = 10, "random"
    # eval_results = baseline_cf(dataset_name, data, metrics, y_cf, model, device, num_rounds=num_rounds, type=type)
    
    if hasattr(data, 'y_cf'):
        y_cf = data.y_cf
    else:
        y_cf = 1 - data.y
    eval_results, edge_mask = compute_counterfactual(dataset_name, data, metrics, y_cf, graphcfe_model, model, device)
    results_all_exp = {}
    for k in metrics:
        results_all_exp = add_list_in_dict(k, results_all_exp, eval_results[k].detach().cpu().numpy())
    for k in eval_results:
        if isinstance(eval_results[k], list):
            print(k, ": ", eval_results[k])
        else:
            print(k, f": {eval_results[k]:.4f}")
    return edge_mask, None

def function_with_args_and_default_kwargs(dict_args, optional_args=None):
    parser = argparse.ArgumentParser()
    # add some arguments
    # add the other arguments
    for k, v in dict_args.items():
        parser.add_argument('--' + k, default=v)
    # args = parser.parse_args(optional_args)
    return parser

def explain_gflowexplainer_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    seed = kwargs["seed"]
    # hidden_dim = kwargs["hidden_dim"]
    gflowexplainer = GFlowExplainer(model, device)
    subdir = os.path.join(kwargs["model_save_dir"], "gflowexplainer")
    os.makedirs(subdir, exist_ok=True)
    gflowexplainer_saving_path = os.path.join(subdir, f"gflowexplainer_{dataset_name}_{str(device)}_{seed}.pickle")
    parser = function_with_args_and_default_kwargs(dict_args=kwargs, optional_args=None)
    train_params = gflow_parse_args(parser)
    train_params.n_hidden = kwargs["hidden_dim"]
    train_params.n_input = kwargs["num_node_features"]
    if os.path.isfile(gflowexplainer_saving_path):
        print("Load saved GFlowExplainer model...")
        gflowexplainer_model = dill.load(open(gflowexplainer_saving_path, "rb"))
        # gflowexplainer_model = gflowexplainer_model.to(device)
    else:
        train_size = min(len(kwargs["dataset"]), 500)
        explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
        explain_dataset = kwargs["dataset"][explain_dataset_idx]
        t0 = time.time()
        gflowexplainer_model = gflowexplainer.train_explainer(train_params, explain_dataset, **kwargs)
        train_time = time.time() - t0
        print("Save GFlowExplainer model...")
        # Save the file
        dill.dump(gflowexplainer_model, file = open(gflowexplainer_saving_path, "wb"))
        train_time_file = os.path.join(subdir, f"gflowexplainer_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)

    # foward_multisteps - origin of this function?
    _, edge_mask = gflowexplainer_model.foward_multisteps(data, gflowexplainer.model)
    # convert removal priority into importance score: rank 3 --> importance score 3/num_edges
    edge_mask = edge_mask/len(edge_mask)
    # edge_mask[i]: indicate the edge of the i-th removal
    # edge_mask = [0,6,3,2,5,4,1] --> [0,1] should be removed first (rank 0), [6,0] should be removed second (rank 1)
    # edge_index = [[0,0,2,3,4,5,6], [1,2,3,4,5,6,0]]
    return edge_mask, None

def explain_rcexplainer_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    seed = kwargs["seed"]
    rcexplainer = RCExplainer_Batch(model, device, kwargs['num_classes'], hidden_size=kwargs['hidden_dim'])
    subdir = os.path.join(kwargs["model_save_dir"], "rcexplainer")
    os.makedirs(subdir, exist_ok=True)
    rcexplainer_saving_path = os.path.join(subdir, f"rcexplainer_{dataset_name}_{str(device)}_{seed}.pickle")
    if os.path.isfile(rcexplainer_saving_path):
        print("Load saved RCExplainer model...")
        rcexplainer_model = dill.load(open(rcexplainer_saving_path, "rb"))
        rcexplainer_model = rcexplainer_model.to(device)
    else:
       # data loader
        train_size = min(len(kwargs["dataset"]), 500)
        explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
        explain_dataset = kwargs["dataset"][explain_dataset_idx]
        dataloader_params = {
            "batch_size": kwargs["batch_size"],
            "random_split_flag": kwargs["random_split_flag"],
            "data_split_ratio": kwargs["data_split_ratio"],
            "seed": kwargs["seed"],
        }
        loader, train_dataset, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)
        t0 = time.time()
        lr, weight_decay, topk_ratio = 0.01, 1e-5, 1.0
        rcexplainer_model = train_rcexplainer(rcexplainer, train_dataset, test_dataset, loader, dataloader_params['batch_size'], lr, weight_decay, topk_ratio)
        train_time = time.time() - t0
        print("Save RCExplainer model...")
        dill.dump(rcexplainer_model, file = open(rcexplainer_saving_path, "wb"))
        train_time_file = os.path.join(subdir, f"rcexplainer_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
    
    max_budget = data.num_edges
    state = torch.zeros(max_budget, dtype=torch.bool)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    edge_ranking = test_policy(rcexplainer_model, model, data, device)
    edge_mask = 1 - edge_ranking/len(edge_ranking)
    # edge_mask[i]: indicate the i-th edge to be added in the search process, i.e. that gives the highest reward.
    return edge_mask, None
 
    
    
def explain_diffexplainer_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    seed = kwargs["seed"]
    diffexplainer = DiffExplainer(model, device)
    
    subdir = os.path.join(kwargs["model_save_dir"], "diffexplainer")
    os.makedirs(subdir, exist_ok=True)
    diffexplainer_saving_path = os.path.join(subdir, f"diffexplainer_{dataset_name}_{str(device)}_{seed}.pth")

    parser = function_with_args_and_default_kwargs(dict_args=kwargs, optional_args=None)
    train_params = diff_parse_args(parser)
    train_params.n_hidden = kwargs["hidden_dim"]
    train_params.feature_in = kwargs["num_node_features"]
    train_params.noise_list = None
    train_params.root = subdir

    if os.path.isfile(diffexplainer_saving_path):
        print("Load saved DiffExplainer model...")
        diffexplainer  = torch.load(diffexplainer_saving_path)
    else:
        # data loader
        train_size = min(len(kwargs["dataset"]), 500)
        explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
        explain_dataset = kwargs["dataset"][explain_dataset_idx]
        dataloader_params = {
            "batch_size": kwargs["batch_size"],
            "random_split_flag": kwargs["random_split_flag"],
            "data_split_ratio": kwargs["data_split_ratio"],
            "seed": kwargs["seed"],
        }
        loader, train_dataset, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)

        t0 = time.time()
        diffexplainer.explain_graph_task(train_params, train_dataset, test_dataset)
        train_time = time.time() - t0

        print("Save DiffExplainer model...")
        torch.save(diffexplainer, diffexplainer_saving_path)
        
        train_time_file = os.path.join(subdir, f"diffexplainer_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)

    # foward_multisteps - origin of this function?
    data.num_graphs = 1
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    explanatory_subgraph = diffexplainer.explanation_generate(train_params, data)
    cmn_edge_idx, cmn_edges, cmn_edge_weight = get_cmn_edges(explanatory_subgraph.edge_index, explanatory_subgraph.edge_weight.cpu().detach().numpy(), data.edge_index)
    edge_mask = cmn_edge_weight
    return edge_mask, None


def explain_gsat_graph(model, data, target, device, **kwargs):
    dataset_name = kwargs["dataset_name"]
    seed = kwargs["seed"]
    num_class = kwargs["num_classes"]

    subdir = os.path.join(kwargs["model_save_dir"], "gsat")
    os.makedirs(subdir, exist_ok=True)
    gsat_saving_path = os.path.join(subdir, f"gsat_{dataset_name}_{str(device)}_{seed}.pt")

    # config gsat training
    shared_config, method_config = gsat_get_config()
    multi_label = False
    extractor = ExtractorMLP(kwargs['hidden_dim'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)
    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    # writer = Writer(log_dir=subdir)
    # hparam_dict = {"dataset": dataset_name, "seed": seed, "device": str(device), "model": kwargs['model_name']}
    metric_dict = deepcopy(init_metric_dict)
    # writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    # data loader
    train_size = min(len(kwargs["dataset"]), 500)
    explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
    explain_dataset = kwargs["dataset"][explain_dataset_idx]
    dataloader_params = {
        "batch_size": kwargs["batch_size"],
        "random_split_flag": kwargs["random_split_flag"],
        "data_split_ratio": kwargs["data_split_ratio"],
        "seed": kwargs["seed"],
    }
    loader, train_dataset, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)


    if os.path.isfile(gsat_saving_path):
        print("Load saved GSAT model...")
        load_checkpoint(extractor, subdir, model_name=f'gsat_{dataset_name}_{str(device)}_{seed}')
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name, num_class, multi_label, seed, method_config, shared_config)
    else:
        print('====================================')
        print('[INFO] Training GSAT...')
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name, num_class, multi_label, seed, method_config, shared_config)
        t0 = time.time()
        metric_dict = gsat.train(loader, test_dataset, metric_dict, use_edge_attr=True)
        train_time = time.time() - t0
        # writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
        save_checkpoint(gsat.extractor, subdir, model_name=f'gsat_{dataset_name}_{str(device)}_{seed}')
        
        train_time_file = os.path.join(subdir, f"gsat_train_time.json")
        entry = {"dataset": dataset_name, "train_time": train_time, "seed": seed, "device": str(device)}
        write_to_json(entry, train_time_file)
        

    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    edge_att, loss_dict, clf_logits = gsat.eval_one_batch(data, epoch=method_config['epochs'])
    edge_mask = edge_att # attention scores
    return edge_mask, None

    