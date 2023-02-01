import os
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam, Saliency
from gnn.model import GCNConv, GATConv, GINEConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from utils.gen_utils import sample_large_graph

from explainer.gnnexplainer import GNNExplainer, TargetedGNNExplainer
from explainer.gnnlrp import GNN_LRP
from explainer.graphsvx import LIME, SHAP, GraphLIME, GraphSVX
from explainer.pgexplainer import PGExplainer
from explainer.pgmexplainer import Node_Explainer
from explainer.subgraphx import SubgraphX
from explainer.zorro import Zorro


def balance_mask_undirected(edge_mask, edge_index):
    balanced_edge_mask = np.zeros(len(edge_mask))
    num_edges = edge_index.shape[1]
    list_edges = edge_index.t().tolist()
    for i, (u, v) in enumerate(list_edges):
        if u > v:
            indices = [
                idx
                for idx in range(num_edges)
                if all(ele in list_edges[idx] for ele in [u, v])
            ]
            if len(indices) == 2:
                balanced_edge_mask[indices] = np.max(
                    [edge_mask[indices[0]], edge_mask[indices[1]]]
                )
    return balanced_edge_mask


def mask_to_directed(edge_mask, edge_index):
    directed_edge_mask = edge_mask.copy()
    for i, (u, v) in enumerate(edge_index.t().tolist()):
        if u > v:
            directed_edge_mask[i] = 0
    return directed_edge_mask


def model_forward_node(x, model, edge_index, edge_attr, node_idx):
    out = model(x, edge_index, edge_attr=edge_attr)
    return out[[node_idx]]


def node_attr_to_edge(edge_index, node_mask):
    edge_mask = np.zeros(edge_index.shape[1])
    edge_mask += node_mask[edge_index[0].cpu().numpy()]
    edge_mask += node_mask[edge_index[1].cpu().numpy()]
    return edge_mask


def get_all_convolution_layers(model):
    layers = []
    for module in model.modules():
        if (
            isinstance(module, GCNConv)
            or isinstance(module, GATConv)
            or isinstance(module, GINEConv)
        ):
            layers.append(module)
    return layers


#### Baselines ####
def explain_random_node(model, data, node_idx, target, device, **kwargs):
    edge_mask = np.random.uniform(size=data.edge_index.shape[1])
    node_feat_mask = np.random.uniform(size=data.x.shape[1])
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_distance_node(model, data, node_idx, target, device, **kwargs):
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    g = to_networkx(data)
    length = nx.shortest_path_length(g, target=node_idx)

    def get_attr(node):
        if node in length:
            return 1 / (length[node] + 1)
        return 0

    edge_sources = data.edge_index[1].cpu().numpy()
    return np.array([get_attr(node) for node in edge_sources]).astype("float"), None


def explain_pagerank_node(model, data, node_idx, target, device, **kwargs):
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    g = to_networkx(data)
    pagerank = nx.pagerank(g, personalization={node_idx: 1})

    node_attr = np.zeros(data.x.shape[0])
    for node, value in pagerank.items():
        node_attr[node] = value
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), None


def explain_basic_gnnexplainer_node(model, data, node_idx, target, device, **kwargs):
    explainer = GNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        epochs=1000,
        edge_ent=kwargs["edge_ent"],
        edge_size=kwargs["edge_size"],
    )
    _, edge_mask = explainer.explain_node(
        node_idx, x=data.x, edge_index=data.edge_index
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


#### Methods ####


def explain_gradcam_node(model, data, node_idx, target, device, **kwargs):
    # Captum default implementation of LayerGradCam does not average over nodes for different channels because of
    # different assumptions on tensor shapes
    input_mask = data.x.clone().requires_grad_(True).to(device)
    layers = get_all_convolution_layers(model)
    node_attrs = []
    for layer in layers:
        layer_gc = LayerGradCam(model_forward_node, layer)
        node_attr = layer_gc.attribute(
            input_mask,
            target=target,
            additional_forward_args=(
                model,
                data.edge_index,
                data.edge_attr,
                node_idx,
            ),
        )
        node_attrs.append(node_attr.squeeze().cpu().detach().numpy())
    node_attr = np.array(node_attrs).mean(axis=0)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), None


def explain_sa_node(model, data, node_idx, target, device, **kwargs):
    saliency = Saliency(model_forward_node)
    input_mask = data.x.clone().requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(
        input_mask,
        target=target,
        additional_forward_args=(model, data.edge_index, data.edge_attr, node_idx),
        abs=False,
    )
    # 1 node feature mask per node.
    node_feat_mask = saliency_mask.cpu().numpy()
    node_attr = node_feat_mask.sum(axis=1)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_ig_node(model, data, node_idx, target, device, **kwargs):
    ig = IntegratedGradients(model_forward_node)
    input_mask = data.x.clone().requires_grad_(True).to(device)
    ig_mask = ig.attribute(
        input_mask,
        target=target,
        additional_forward_args=(model, data.edge_index, data.edge_attr, node_idx),
        internal_batch_size=input_mask.shape[0],
    )
    node_feat_mask = ig_mask.cpu().detach().numpy()
    node_attr = node_feat_mask.sum(axis=1)
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_occlusion_node(model, data, node_idx, target, device, **kwargs):
    depth_limit = kwargs["num_layers"] + 1
    data = Data(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
    if target is None:
        pred_probs = (
            model(data.x, data.edge_index, data.edge_attr)[node_idx]
            .cpu()
            .detach()
            .numpy()
        )
        pred_prob = pred_probs.max()
        target = pred_probs.argmax()
    else:
        pred_prob = 1
    g = to_networkx(data)
    subgraph_nodes = []
    for k, v in nx.shortest_path_length(g, target=node_idx).items():
        if v < depth_limit:
            subgraph_nodes.append(k)
    subgraph = g.subgraph(subgraph_nodes)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        # if include_edges is not None and not include_edges[i].item():
        # continue
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in subgraph.edges():
            edge_occlusion_mask[i] = False
            prob = model(
                data.x,
                data.edge_index[:, edge_occlusion_mask],
                data.edge_attr[edge_occlusion_mask],
            )[node_idx][target].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask.astype("float"), None


def gpu_to_cpu(data, device):
    data.x = torch.FloatTensor(data.x.cpu().numpy().copy()).to(device)
    data.edge_index = torch.LongTensor(data.edge_index.cpu().numpy().copy()).to(device)
    data.edge_attr = torch.FloatTensor(data.edge_attr.cpu().numpy().copy()).to(device)
    return data


def explain_gnnexplainer_node(model, data, node_idx, target, device, **kwargs):
    data = gpu_to_cpu(data, device)
    explainer = TargetedGNNExplainer(
        model,
        num_hops=kwargs["num_layers"],
        epochs=1000,
        edge_ent=kwargs["edge_ent"],
        edge_size=kwargs["edge_size"],
        allow_edge_mask=True,
        allow_node_mask=True,
        device=device,
    )
    node_feat_mask, edge_mask = explainer.explain_node_with_target(
        node_idx,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        target=target,
    )
    edge_mask = edge_mask.cpu().detach().numpy()
    # 1 node feature mask for all the nodes.
    node_feat_mask = node_feat_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_pgmexplainer_node(model, data, node_idx, target, device, **kwargs):
    explainer = Node_Explainer(
        model,
        data.edge_index,
        data.edge_attr,
        data.x,
        kwargs["num_layers"],
        device=device,
        print_result=0,
    )
    explanation = explainer.explain(
        node_idx,
        target,
        num_samples=100,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1,
    )
    node_attr = np.zeros(data.x.shape[0])
    for node, p_value in explanation.items():
        node_attr[node] = 1 - p_value
    edge_mask = node_attr_to_edge(data.edge_index, node_attr)
    return edge_mask.astype("float"), None


def explain_subgraphx_node(model, data, node_idx, target, device, **kwargs):
    subgraphx = SubgraphX(
        model,
        kwargs["num_classes"],
        device,
        num_hops=kwargs["num_layers"],
        explain_graph=False,
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
        node_idx=node_idx,
    )
    return edge_mask.astype("float"), None


def explain_zorro_node(model, data, node_idx, target, device, **kwargs):
    zorro = Zorro(model, device, num_hops=kwargs["num_layers"])
    print("explain node", zorro.explain_node(node_idx, data.x, data.edge_index))
    explanation = zorro.explain_node(
        node_idx, data.x, data.edge_index, tau=0.85, recursion_depth=3
    )
    print("explanation", explanation)
    # selected_nodes, selected_features, executed_selection = zorro.explain_node(node_idx, x, edge_index, tau=0.85, recursion_depth=4)
    # selected_nodes = torch.Tensor(selected_nodes.squeeze())
    # selected_features = torch.Tensor(selected_features.squeeze())
    # print("node_attrs", selected_nodes)
    # print("node_feature_mask", selected_features)
    # print("executed_selection", executed_selection)
    # node_attr = np.array(selected_nodes)
    # edge_mask = node_attr_to_edge(edge_index, node_attr)
    # edge_mask = edge_mask.cpu().detach().numpy()
    # node_feature_mask = node_feature_mask.cpu().detach().numpy()
    return  # edge_mask.astype("float"), node_feature_mask.astype("float")


def explain_pgexplainer_node(model, data, node_idx, target, device, **kwargs):
    pgexplainer = PGExplainer(
        model,
        in_channels=kwargs["hidden_dim"] * 3,
        device=device,
        num_hops=kwargs["num_layers"],
        explain_graph=False,
    )
    dataset_name = kwargs["dataset_name"]
    subdir = os.path.join(kwargs["model_save_dir"], "pgexplainer")
    os.makedirs(subdir, exist_ok=True)
    pgexplainer_saving_path = os.path.join(subdir, f"pgexplainer_{dataset_name}.pth")
    if os.path.isfile(pgexplainer_saving_path):
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    else:
        data = sample_large_graph(data)
        pgexplainer.train_explanation_network(data)
        print("Save PGExplainer model...")
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    edge_mask = pgexplainer.explain_node(node_idx, data.x, data.edge_index)
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_gnnlrp_node(model, data, node_idx, target, device, **kwargs):
    gnnlrp = GNN_LRP(model)
    walks, edge_mask = gnnlrp(data.x, data.edge_index)
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_graphsvx_node(model, data, node_idx, target, device, **kwargs):
    graphsvx = GraphSVX(data, model, device)
    node_feat_mask = graphsvx.explain(node_indexes=[node_idx], multiclass=False)
    coefs = node_feat_mask[0].T[graphsvx.F :]
    print(coefs)
    print(coefs.shape)
    print("node_feat_mask", node_feat_mask[0])
    print("node_feat_mask shape", node_feat_mask[0].shape)
    print(graphsvx.base_values)
    print(graphsvx.base_values.shape)
    return None, node_feat_mask.astype("float")


def explain_graphlime_node(model, data, node_idx, target, device, **kwargs):
    graphlime = GraphLIME(data, model, device, hop=2, rho=0.1, cached=True)
    node_feat_mask = graphlime.explain(
        node_idx, hops=None, num_samples=None, info=False, multiclass=False
    )
    return None, node_feat_mask.astype("float")


def explain_lime_node(model, data, node_idx, target, device, **kwargs):
    graphlime = LIME(data, model, device)
    node_feat_mask = graphlime.explain(
        node_idx, hops=None, num_samples=10, info=False, multiclass=False
    )
    return None, node_feat_mask.astype("float")


def explain_shap_node(model, data, node_idx, target, device, **kwargs):
    ":return: shapley values for features that influence node v's pred"
    shap = SHAP(data, model, device)
    node_feat_mask = shap.explain()
    return None, node_feat_mask.astype("float")
