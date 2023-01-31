import numpy as np
import torch
import networkx as nx
import numpy as np
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, download_url
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx
from dataset.syn_utils.gengraph import *


class SynGraphDataset:
    r"""
    The Synthetic datasets used in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.
    It takes Barabási–Albert(BA) graph or balance tree as base graph
    and randomly attachs specific motifs to the base graph.
    Args:
        root (:obj:`str`): Root data directory to save datasets
        name (:obj:`str`): The name of the dataset. Including :obj:`BA_shapes`, BA_grid,
        transform (:obj:`Callable`, :obj:`None`): A function/transform that takes in an
            :class:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (:obj:`Callable`, :obj:`None`):  A function/transform that takes in
            an :class:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    # Format: name: [display_name, url_name, filename]
    names = {
        "ba_house": ["BA_House", "BA_House.pkl", "BA_House"],
        "ba_grid": ["BA_Grid", "BA_Grid.pkl", "BA_Grid"],
        "ba_bottle": ["BA_bottle", "BA_bottle.pkl", "BA_bottle"],
        "ba_community": ["BA_Community", "BA_Community.pkl", "BA_Community"],
        "tree_grid": ["Tree_Grid", "Tree_Grid.pkl", "Tree_Grid"],
        "tree_cycle": ["Tree_Cycle", "Tree_Cycles.pkl", "Tree_Cycles"],
        "ba_2motifs": ["BA_2Motifs", "BA_2Motifs.pkl", "BA_2Motifs"],
    }

    def __init__(self, root, name, **dataset_params):
        self.root = root
        self.name = name.lower()
        self.data = None
        self.dataset_params = dataset_params

    def dir(self):
        return osp.join(self.root, self.name)

    def file_names(self):
        return osp.join(
            self.dir(),
            "{}_{}_{}.pt".format(
                self.name,
                self.dataset_params["num_shapes"],
                self.dataset_params["width_basis"],
            ),
        )

    def process(self):
        if not os.path.isfile(self.file_names()):
            os.makedirs(self.dir(), exist_ok=True)
            data = self.build_syndata()
            torch.save(data, self.file_names())
        data = torch.load(self.file_names())
        self.data = data
        return

    def __repr__(self):
        return "{}({}, {})".format(
            self.names[self.name][0],
            self.dataset_params["num_shapes"],
            self.dataset_params["width_basis"],
        )

    def build_syndata(self):
        """Generate synthetic graohs and convert them into Pytorch geometric Data object.

        Returns:
            Data: converted synthetic Pytorch geometric Data object
        """
        generate_function = "gen_" + self.name

        G, labels, name = eval(generate_function)(
            nb_shapes=self.dataset_params["num_shapes"],
            width_basis=self.dataset_params["width_basis"],
            feature_generator=featgen.ConstFeatureGen(
                np.ones(self.dataset_params["num_node_features"], dtype=float)
            ),
        )

        data = from_networkx(G.to_undirected(), all)
        data.adj = torch.LongTensor(nx.to_numpy_matrix(G))
        data.num_classes = len(np.unique(labels))
        data.y = torch.LongTensor(labels)
        data.x = data.x.float()
        data.edge_attr = torch.ones(data.edge_index.size(1))
        n = data.num_nodes
        data.train_mask, data.val_mask, data.test_mask = (
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.bool),
        )
        train_ids, test_ids = train_test_split(
            range(n),
            test_size=self.dataset_params["test_ratio"],
            random_state=self.dataset_params["seed"],
            shuffle=True,
        )
        train_ids, val_ids = train_test_split(
            train_ids,
            test_size=self.dataset_params["val_ratio"],
            random_state=self.dataset_params["seed"],
            shuffle=True,
        )

        data.train_mask[train_ids] = 1
        data.val_mask[val_ids] = 1
        data.test_mask[test_ids] = 1

        return [data]
