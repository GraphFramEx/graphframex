import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerGradCam, Saliency
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from utils.gen_utils import from_adj_to_edge_index, from_edge_index_to_adj

from explainer.gnnexplainer import TargetedGNNExplainer
from explainer.pgmexplainer import Graph_Explainer
from explainer.subgraphx import SubgraphX


def model_forward_graph(adj, model, x):
    out, _ = model.forward_adj(x, adj)
    return out


def node_attr_to_edge(edge_index, node_mask):
    edge_mask = np.zeros(edge_index.shape[1])
    edge_mask += node_mask[edge_index[0].cpu().numpy()]
    edge_mask += node_mask[edge_index[1].cpu().numpy()]
    return edge_mask


#### Baselines ####
def explain_random_graph(
    model, x, edge_index, target, device, args, include_edges=None
):
    return np.random.uniform(size=edge_index.shape[1])


def explain_sa_graph(model, x, edge_index, target, device, args, include_edges=None):
    saliency = Saliency(model_forward_graph)
    edge_weights = torch.ones(edge_index.size(1))
    max_n = x.size(0)
    adj = from_edge_index_to_adj(edge_index, edge_weights, max_n)
    adj_mask = adj.requires_grad_(True).to(device)
    saliency_mask = saliency.attribute(
        adj_mask, target=int(target), additional_forward_args=(model, x), abs=False
    )
    edge_att = saliency_mask * adj
    edge_att = pow(edge_att, 2).cpu().detach().numpy()
    edges, edge_mask = from_adj_to_edge_index(edge_att)
    edge_mask = norm_imp(edge_mask.cpu().detach().numpy())
    return edge_mask


def norm_imp(imp):
    imp[imp < 0] = 0
    imp += 1e-16
    return imp / imp.sum()


"""
def explain_sa_graph(model, x, edge_index, target, device, args, include_edges=None):
    edge_weights = Variable(torch.ones(edge_index.shape[1]), requires_grad=True)
    x = Variable(x, requires_grad=True)
    max_n = x.size(0)
    adj_init = from_edge_index_to_adj(edge_index, torch.FloatTensor(edge_weights), max_n)
    adj = Variable(adj_init, requires_grad=True)
    pred, _ = model.forward_adj(x, adj)
    pred[0, target].backward()
    edge_att = adj.grad * adj_init
    edges, edge_grads = from_adj_to_edge_index(edge_att)
    edge_mask = pow(edge_grads, 2).cpu().detach().numpy()
    edge_mask = norm_imp(edge_mask)
    return edge_mask"""


def explain_ig_graph(model, x, edge_index, target, device, args, include_edges=None):
    ig = IntegratedGradients(model_forward_graph)
    edge_weights = torch.ones(edge_index.size(1))
    max_n = x.size(0)
    adj = from_edge_index_to_adj(edge_index, edge_weights, max_n)
    adj_mask = adj.requires_grad_(True).to(device)
    ig_mask = ig.attribute(
        adj_mask,
        target=int(target),
        additional_forward_args=(model, x),
        internal_batch_size=adj_mask.shape[1],
    )
    edge_att = ig_mask * adj
    edge_att = pow(edge_att, 2).cpu().detach().numpy()
    edges, edge_mask = from_adj_to_edge_index(edge_att)
    edge_mask = norm_imp(edge_mask.cpu().detach().numpy())
    return edge_mask


def explain_gradcam_graph(
    model, x, edge_index, target, device, args, include_edges=None
):
    edge_weights = Variable(torch.ones(edge_index.shape[1]), requires_grad=True)
    x = Variable(x, requires_grad=True)
    max_n = x.size(0)
    adj_init = from_edge_index_to_adj(
        edge_index, torch.FloatTensor(edge_weights), max_n
    )
    adj = Variable(adj_init, requires_grad=True)
    pred, _ = model.forward_adj(x, adj)
    pred[0, target].backward()
    edge_att = adj.grad * adj_init
    edges, edge_grads = from_adj_to_edge_index(edge_att)
    alpha = torch.mean(edge_grads)
    edge_mask = F.relu(edge_weights * alpha).cpu().detach().numpy()
    edge_mask = norm_imp(edge_mask)
    return edge_mask


def explain_occlusion_graph(
    model, x, edge_index, target, device, args, include_edges=None
):
    depth_limit = args.num_gc_layers + 1
    data = Data(x=x, edge_index=edge_index)
    if target is None:
        pred_probs = model(x, edge_index).cpu().detach().numpy()
        pred_prob = pred_probs[target]
    else:
        pred_prob = 1
    g = to_networkx(data)
    edge_occlusion_mask = np.ones(data.num_edges, dtype=bool)
    edge_mask = np.zeros(data.num_edges)
    edge_index_numpy = data.edge_index.cpu().numpy()
    for i in range(data.num_edges):
        if include_edges is not None and not include_edges[i].item():
            continue
        u, v = list(edge_index_numpy[:, i])
        if (u, v) in g.edges():
            edge_occlusion_mask[i] = False
            prob = model(data.x, data.edge_index[:, edge_occlusion_mask])[0][
                target
            ].item()
            edge_mask[i] = pred_prob - prob
            edge_occlusion_mask[i] = True
    return edge_mask


def explain_gnnexplainer_graph(
    model, x, edge_index, target, device, args, include_edges=None, **kwargs
):
    if "edge_weights" not in kwargs:
        kwargs["edge_weights"] = None

    explainer = TargetedGNNExplainer(
        model,
        num_hops=args.num_gc_layers,
        epochs=args.num_epochs,
        edge_ent=args.edge_ent,
        edge_size=args.edge_size,
        allow_node_mask=False,
    )
    if eval(args.explain_graph):
        edge_mask = explainer.explain_graph_with_target(
            x=x,
            edge_index=edge_index,
            edge_weights=kwargs["edge_weights"],
            target=target,
        )

    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask


def explain_pgmexplainer_graph(
    model, x, edge_index, target, device, args, include_edges=None
):
    explainer = Graph_Explainer(model, edge_index, x, device=device, print_result=0)
    explanation = explainer.explain(
        num_samples=1000,
        percentage=10,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1,
    )
    node_attr = np.zeros(x.shape[0])
    for node, p_value in explanation.items():
        node_attr[node] = 1 - p_value
    edge_mask = node_attr_to_edge(edge_index, node_attr)
    return edge_mask


def explain_subgraphx_graph(
    model, x, edge_index, target, device, args, include_edges=None
):
    subgraphx = SubgraphX(
        model, args.num_classes, device, num_hops=2, explain_graph=True
    )
    edge_mask = subgraphx.explain(
        x, edge_index, max_nodes=args.num_top_edges, label=target
    )
    return edge_mask
