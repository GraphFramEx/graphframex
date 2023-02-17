import os
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from code.explainer.gnnlrp import GNN_LRP
from code.explainer.pgexplainer import PGExplainer
from code.utils.math_utils import sigmoid
from utils.gen_utils import (
    from_adj_to_edge_index,
    from_edge_index_to_adj,
    sample_large_graph,
)
from gnn.model import GCNConv, GATConv, GINEConv, TransformerConv

from explainer.gnnexplainer import TargetedGNNExplainer
from explainer.pgmexplainer import Graph_Explainer
from explainer.subgraphx import SubgraphX
from explainer.gradcam import GraphLayerGradCam


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


def explain_basic_gnnexplainer_node(model, data, target, device, **kwargs):
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
    pgexplainer_saving_path = os.path.join(subdir, f"pgexplainer_{dataset_name}.pth")
    if os.path.isfile(pgexplainer_saving_path):
        print("Load saved PGExplainer model...")
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    else:
        data = sample_large_graph(data)
        pgexplainer.train_explanation_network(kwargs["dataset"][:200])
        print("Save PGExplainer model...")
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
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
