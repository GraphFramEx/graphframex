from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from torch_geometric.utils.convert import to_networkx

from utils.io_utils import check_dir, gen_mask_density_plt_name


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


def plot_mask_density(edge_mask, args):
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    sns.histplot(edge_mask, kde=True, ax=ax)

    plt.xlim(0, 1)
    plt.title(
        f"Density of edge mask for {args.explainer_name}, entropy = {args.edge_ent}, mask size = {args.edge_size}"
    )
    plt.xlabel("edge importance")
    print(gen_mask_density_plt_name(args))
    plt.savefig(gen_mask_density_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")


# def plot_explanation(data, edge_masks):


def plot_expl_nc(G, G_true, role, node_idx, args, top_acc):

    G = G.to_undirected()
    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    nodes, labels = zip(*nx.get_node_attributes(G, "label").items())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    nx.draw(
        G_true.to_undirected(),
        cmap=plt.get_cmap("tab10"),
        with_labels=True,
        node_color=role,
        font_weight="bold",
        vmin=0,
        vmax=3,
        ax=ax1,
    )
    nx.draw(
        G,
        cmap=plt.get_cmap("tab10"),
        with_labels=True,
        node_color=labels,
        font_weight="bold",
        vmin=0,
        vmax=3,
        edgelist=edges,
        edge_color=weights,
        width=2,
        edge_cmap=plt.cm.Blues,
        ax=ax2,
    )
    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    check_dir(f"figures/{args.dataset}/")
    plt.savefig(
        f"figures/{args.dataset}/fig_expl_nc_hard_top_{top_acc}_{args.hard_mask}_{args.dataset}_{args.explainer_name}_{node_idx}_{date}.pdf"
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
