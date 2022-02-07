import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.io_utils import gen_mask_density_plt_name


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
