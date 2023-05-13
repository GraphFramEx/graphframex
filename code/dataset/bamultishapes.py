import pickle
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from networkx.generators import random_graphs, lattice, small, classic
import networkx as nx
import pickle as pkl
import random
import os
import os.path as osp
from networkx.algorithms.operators.binary import compose, union
from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    k_hop_subgraph,
    to_scipy_sparse_matrix,
    to_dense_adj,
    subgraph,
)

# from utils.gen_utils import padded_datalist, from_edge_index_to_adj


class BAMultiShapesDataset(InMemoryDataset):
    r"""The synthetic BA-Multi-Shapes graph classification dataset for
    evaluating explainabilty algorithms, as described in the
    `"Global Explainability of GNNs via Logic Combination of Learned Concepts"
    <https://arxiv.org/abs/2210.07147>`_ paper.
    Given three atomic motifs, namely House (H), Wheel (W), and Grid (G),
    :class:`~torch_geometric.datasets.BAMultiShapesDataset` contains 1,000
    graphs where each graph is obtained by attaching the motifs to a random
    Barabasi-Albert (BA) as follows:

    * class 0: :math:`\emptyset \lor H \lor W \lor G \lor \{ H, W, G \}`

    * class 1: :math:`(H \land W) \lor (H \land G) \lor (W \land G)`

    This dataset is pre-computed from the official implementation.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - #graphs
          - #nodes
          - #edges
          - #features
          - #classes
        * - 1000
          - 40
          - ~87.0
          - 10
          - 2
    """
    url = (
        "https://github.com/steveazzolin/gnn_logic_global_expl/raw/master/"
        "datasets/BAMultiShapes/BAMultiShapes.pkl"
    )

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return "BAMultiShapes.pkl"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        # download_url(self.url, self.raw_dir)
        if not osp.exists(osp.join(self.raw_dir, file)):
            print(
                "raw data of `{}` doesn't exist, please download from our github.".format(
                    file
                )
            )
            raise FileNotFoundError

    def process(self):
        with open(self.raw_paths[0], "rb") as f:
            adjs, xs, ys, e_labels = pickle.load(f)

        data_list: List[Data] = []
        adj_list = []
        max_num_nodes = 0
        index = 0
        for adj, x, y, e_label in zip(adjs, xs, ys, e_labels):
            edge_index = torch.from_numpy(adj).nonzero().t()
            x = torch.from_numpy(np.array(x)).to(torch.float)
            edge_label = torch.from_numpy(
                np.array([val for val in e_label.values()])
            ).to(torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                edge_label=torch.cat([edge_label, edge_label], dim=0),
                idx=index,
            )
            index += 1

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            max_num_nodes = max(max_num_nodes, data.num_nodes)
            adj = from_edge_index_to_adj(data.edge_index, None, data.num_nodes)

            adj_list.append(adj)
            data_list.append(data)

        data_list = padded_datalist(data_list, adj_list, max_num_nodes)
        torch.save(self.collate(data_list), self.processed_paths[0])


def merge_graphs(g1, g2, nb_random_edges=1):
    mapping = dict()
    max_node = max(g1.nodes())

    i = 1
    for n in g2.nodes():
        mapping[n] = max_node + i
        i = i + 1
    g2 = nx.relabel_nodes(g2, mapping)

    g12 = nx.union(g1, g2)
    for i in range(nb_random_edges):
        e1 = list(g1.nodes())[np.random.randint(0, len(g1.nodes()))]
        e2 = list(g2.nodes())[np.random.randint(0, len(g2.nodes()))]
        g12.add_edge(e1, e2, edge_label=0)
    return g12


def generate_class1(nb_random_edges, nb_node_ba=40):
    r = np.random.randint(3)

    if r == 0:  # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 6 - 9, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = classic.wheel_graph(6)
        nx.set_edge_attributes(g2, {(u, v): 3 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        nx.set_edge_attributes(g3, {(u, v): 2 for (u, v) in g3.edges()}, "edge_label")
        g123 = merge_graphs(g12, g3, nb_random_edges)
    elif r == 1:  # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 6 - 5, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = classic.wheel_graph(6)
        nx.set_edge_attributes(g2, {(u, v): 3 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
        g3 = small.house_graph()
        nx.set_edge_attributes(g3, {(u, v): 1 for (u, v) in g3.edges()}, "edge_label")
        g123 = merge_graphs(g12, g3, nb_random_edges)
    elif r == 2:  # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 5 - 9, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = small.house_graph()
        nx.set_edge_attributes(g2, {(u, v): 1 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        nx.set_edge_attributes(g3, {(u, v): 2 for (u, v) in g3.edges()}, "edge_label")
        g123 = merge_graphs(g12, g3, nb_random_edges)
    return g123


def generate_class0(nb_random_edges, nb_node_ba=40):
    r = np.random.randint(10)

    if r > 3:
        g12 = random_graphs.barabasi_albert_graph(nb_node_ba, 1)
        nx.set_edge_attributes(g12, {(u, v): 0 for (u, v) in g12.edges()}, "edge_label")
    if r == 0:  # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 6, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = classic.wheel_graph(6)
        nx.set_edge_attributes(g2, {(u, v): 3 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
    if r == 1:  # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 5, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = small.house_graph()
        nx.set_edge_attributes(g2, {(u, v): 1 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
    if r == 2:  # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 9, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = lattice.grid_2d_graph(3, 3)
        nx.set_edge_attributes(g2, {(u, v): 2 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
    if r == 3:  # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba - 9 - 5 - 6, 1)
        nx.set_edge_attributes(g1, {(u, v): 0 for (u, v) in g1.edges()}, "edge_label")
        g2 = small.house_graph()
        nx.set_edge_attributes(g2, {(u, v): 1 for (u, v) in g2.edges()}, "edge_label")
        g12 = merge_graphs(g1, g2, nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        nx.set_edge_attributes(g3, {(u, v): 2 for (u, v) in g3.edges()}, "edge_label")
        g123 = merge_graphs(g12, g3, nb_random_edges)
        g4 = classic.wheel_graph(6)
        nx.set_edge_attributes(g4, {(u, v): 3 for (u, v) in g4.edges()}, "edge_label")
        g12 = merge_graphs(g123, g4, nb_random_edges)
    return g12


def generate(num_samples):
    assert num_samples % 2 == 0
    adjs = []
    labels = []
    feats = []
    edge_labels = []
    nb_node_ba = 40

    for _ in range(int(num_samples / 2)):
        g = generate_class1(nb_random_edges=1, nb_node_ba=nb_node_ba)
        adjs.append(nx.adjacency_matrix(g).A)
        labels.append(0)
        feats.append(list(np.ones((len(g.nodes()), 10)) / 10))
        edge_labels.append(nx.get_edge_attributes(g, "edge_label"))

    for _ in range(int(num_samples / 2)):
        g = generate_class0(nb_random_edges=1, nb_node_ba=nb_node_ba)
        adjs.append(nx.adjacency_matrix(g).A)
        labels.append(1)
        feats.append(list(np.ones((len(g.nodes()), 10)) / 10))
        edge_labels.append(nx.get_edge_attributes(g, "edge_label"))
    return adjs, feats, labels, edge_labels


def save(data, root):
    path = os.path.join(root, "bamultishapes/raw")
    os.makedirs(path, exist_ok=True)
    f = open(os.path.join(path, "BAMultiShapes.pkl"), "wb")
    pkl.dump(data, f)
    f.close()


def from_edge_index_to_adj(edge_index, edge_weight, max_n):
    adj = to_scipy_sparse_matrix(edge_index, edge_attr=edge_weight).toarray()
    assert len(adj) <= max_n, "The adjacency matrix contains more nodes than the graph!"
    if len(adj) < max_n:
        adj = np.pad(adj, (0, max_n - len(adj)), mode="constant")
    return torch.FloatTensor(adj)


def padded_datalist(data_list, adj_list, max_num_nodes):
    for i, data in enumerate(data_list):
        data.adj_padded = padding_graphs(adj_list[i], max_num_nodes)
        data.x_padded = padding_features(data.x, max_num_nodes)
    return data_list


def padding_graphs(adj, max_num_nodes):
    num_nodes = adj.shape[0]
    adj_padded = np.eye((max_num_nodes))
    adj_padded[:num_nodes, :num_nodes] = adj.cpu()
    return torch.tensor(adj_padded, dtype=torch.long)


def padding_features(features, max_num_nodes):
    feat_dim = features.shape[1]
    num_nodes = features.shape[0]
    features_padded = np.zeros((max_num_nodes, feat_dim))
    features_padded[:num_nodes] = features.cpu()
    return torch.tensor(features_padded, dtype=torch.float)


if __name__ == "__main__":
    root_path_data = "/cluster/work/zhang/kamara/graphframex/data"
    if os.path.exists(
        os.path.join(root_path_data, "bamultishapes/raw/BAMultiShapes.pkl")
    ):
        print("Data already generated")
        dataset = BAMultiShapesDataset(root=root_path_data, name="BAMultiShapes")
        print(dataset.data)
        for data in dataset:
            print(data)
            print(data.edge_index)
            print(data.edge_label)
            break
    else:
        print("Generate data")
        data = generate(1000)
        save(data, root=root_path_data)
