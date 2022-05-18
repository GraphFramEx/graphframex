from datetime import datetime
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from torch_geometric.utils.convert import to_networkx

from utils.io_utils import check_dir, gen_feat_importance_plt_name, gen_mask_density_plt_name

def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
            node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
                    node(s).
            num_hops: (int): The number of hops :math:`k`.
            edge_index (LongTensor): The edge indices.
            relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
                    :obj:`edge_index` will be relabeled to hold consecutive indices
                    starting from zero. (default: :obj:`False`)
            num_nodes (int, optional): The number of nodes, *i.e.*
                    :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
            flow (string, optional): The flow direction of :math:`k`-hop
                    aggregation (:obj:`"source_to_target"` or
                    :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
                     :class:`BoolTensor`)
    """

    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

    

def custom_to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                    remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data.__dict__.items():
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def plot_avg_density(edge_masks, args):
    rank_masks = [np.sort(edge_mask) for edge_mask in edge_masks]
    avg_mask = np.mean(rank_masks, axis=0)

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)
    sns.distplot(avg_mask, kde=True, ax=ax)
    plt.xlim(0, 1)
    plt.title(f"Density of averaged edge mask for {args.explainer_name}")
    plt.xlabel("edge importance")
    plt.savefig(gen_mask_density_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")


def plot_mask_density(mask, args, type="edge"):
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    sns.histplot(mask, kde=True, ax=ax)

    plt.xlim(0, 1)
    plt.title(
        f"Density of {type} mask for {args.explainer_name}, entropy = {args.edge_ent}, mask size = {args.edge_size}"
    )
    plt.xlabel(f"{type} importance")
    print(gen_mask_density_plt_name(args))
    plt.savefig(gen_mask_density_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

def plot_masks_density(masks, args, type="edge"):
    pal = sns.color_palette("tab10")
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    # Plotting avg density
    # edge_values = np.array(edge_masks).reshape(-1)
    # positive_edge_values = edge_values[edge_values>0]
    max_len = 0
    for i in range(5):
        mask = masks[i]
        pos_mask = mask[mask > 0]
        max_len = max_len if max_len > len(pos_mask) else len(pos_mask)
        sns.histplot(pos_mask, kde=True, color=pal[i], alpha=0.4, ax=ax)
    plt.xlim(0, 1)
    plt.ylim(0, max_len)
    plt.title(
        f"Density of 5 {type} masks for {args.explainer_name}, target as true label = {args.true_label_as_target}, sparsity = {args.sparsity}"
    )
    plt.xlabel(f"{type} importance")
    print(gen_mask_density_plt_name(args, type))
    plt.savefig(gen_mask_density_plt_name(args, type), dpi=600)
    plt.close()
    matplotlib.style.use("default")

def plot_feat_importance(node_feat_mask, args):
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    sns.barplot(range(0,len(node_feat_mask)), node_feat_mask, ax=ax)
    plt.xlim(0, 1)
    plt.title(f"Node feature importance for {args.explainer_name}")
    plt.xlabel("Node feature importance")
    plt.savefig(gen_feat_importance_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

def plot_feat_importance(node_feat_masks, args):
    pal = sns.color_palette("tab10")
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    for i in range(5):
        node_feat_mask = node_feat_masks[i]

    sns.barplot(range(0,len(node_feat_mask)), node_feat_mask, ax=ax)
    plt.xlim(0, 1)
    plt.title(f"Node feature importance for {args.explainer_name}")
    plt.xlabel("Node feature importance")
    plt.savefig(gen_feat_importance_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")


# def plot_explanation(data, edge_masks):


def plot_expl_nc(G, G_true, role, node_idx, args, top_acc):
    G = G.to_undirected()
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    nodes, labels = zip(*nx.get_node_attributes(G, "label").items())
    nodes = np.array(nodes)
    labels = np.array(labels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    index_target_node = np.where(nodes==node_idx)[0][0]
    dict_color_labels = {0: 'orange', 1: 'green', 2: 'green', 3:'green'}
    node_labels = [dict_color_labels[l] for l in labels]
    node_labels[index_target_node] = 'tab:red'
    pos=nx.spring_layout(G_true, k=0.05, iterations=20),
    print(pos)
    nx.draw(
        G_true.to_undirected(),
        pos=pos,
        with_labels=False,
        node_color='green',
        edge_color="black",
        width=2,
        ax=ax1,
    )
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_color=node_labels,
        edge_color="black",
        width=2,
        ax=ax2,
    )
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    # dir_name = f"figures/{args.dataset}/node_{node_idx}/{args.explainer_name}/{args.strategy}_{args.param}"
    dir_name = f"figures/{args.dataset}/node_{node_idx}/{args.explainer_name}/"
    check_dir(dir_name)
    plt.savefig(
        ### os.path.join(dir_name, f"fig_expl_nc_top_{top_acc}_sparsity_{args.param}_{args.hard_mask}_{args.dataset}_{args.explainer_name}_{node_idx}_{date}.pdf")
        os.path.join(dir_name, f"fig_expl_nc_top_{top_acc}_{args.dataset}_{args.explainer_name}_{node_idx}_{date}.pdf")
    )


def plot_expl_gc(data_list, edge_masks, args, num_plots=5):
    if args.num_test < num_plots:
        num_plots = args.num_test
    fig, axs = plt.subplots(num_plots, 2, figsize=(15, 10 * num_plots), sharey=True)
    fig.set_dpi(600)
    for i in range(num_plots):
        data = data_list[i]
        atoms = np.argmax(data.x, axis=1)
        G_init = to_networkx(data)
        pos = nx.spring_layout(G_init)
        nx.draw(
            G_init.to_undirected(),
            pos,
            cmap=plt.get_cmap("tab10"),
            node_color=atoms,
            with_labels=True,
            font_weight="bold",
            vmin=0,
            vmax=6,
            ax=axs[i][0],
        )
        k = 0
        for u, v, d in G_init.edges(data=True):
            d["weight"] = edge_masks[i][k]
            k += 1
        G_masked = G_init.copy()
        for u, v, d in G_masked.edges(data=True):
            d["weight"] = (G_init[u][v]["weight"] + G_init[v][u]["weight"]) / 2
        G_masked = G_masked.to_undirected()
        edges, weights = zip(*nx.get_edge_attributes(G_masked, "weight").items())

        nx.draw(
            G_masked,
            pos,
            cmap=plt.get_cmap("tab10"),
            node_color=atoms,
            with_labels=True,
            font_weight="bold",
            vmin=0,
            vmax=6,
            edgelist=edges,
            edge_color=weights,
            width=2,
            edge_cmap=plt.cm.Blues,
            edge_vmin=0,
            edge_vmax=1,
            ax=axs[i][1],
        )

    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    check_dir(f"figures/{args.dataset}/")
    plt.savefig(f"figures/{args.dataset}/fig_expl_gc_hard_{args.hard_mask}_{args.dataset}_{date}.pdf")


def _fruchterman_reingold(
    A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    import numpy as np

    try:
        nnodes, _ = A.shape
    except AttributeError as e:
        msg = "fruchterman_reingold() takes an adjacency matrix as input"
        raise nx.NetworkXError(msg) from e

    if pos is None:
        # random initial positions
        pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    # We need to calculate this in case our fixed positions force our domain
    # to be much bigger than 1x1
    t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for iteration in range(iterations):
        # matrix of difference between points
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        # distance between points
        distance = np.linalg.norm(delta, axis=-1)
        # enforce minimum distance of 0.01
        np.clip(distance, 0.01, None, out=distance)
        # displacement "force"
        displacement = np.einsum(
            "ijk,ij->ik", delta, (k * k / distance ** 2 - A * distance / k)
        )
        # ADD THIS LINE - prevent things from flying off into infinity if not connected
        displacement = displacement - pos / ( k * np.sqrt(nnodes))
        # update positions
        length = np.linalg.norm(displacement, axis=-1)
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.einsum("ij,i->ij", displacement, t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        pos += delta_pos
        # cool temperature
        t -= dt
        err = np.linalg.norm(delta_pos) / nnodes
        if err < threshold:
            break
    return pos