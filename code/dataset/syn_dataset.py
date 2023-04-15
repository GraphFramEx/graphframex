import numpy as np
import torch
import networkx as nx
import pickle
import numpy as np
import os
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from sklearn.model_selection import train_test_split
from torch_geometric.utils import from_networkx
from dataset.syn_utils.gengraph import *
from torch_geometric.utils import dense_to_sparse
from utils.gen_utils import padded_datalist, from_edge_index_to_adj


class SynGraphDataset(InMemoryDataset):
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

    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/{}'
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

    def __init__(self, root, name, transform=None, pre_transform=None, **dataset_params):
        self.name = name.lower()
        self.dataset_params = dataset_params
        super(SynGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.pkl'

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        if self.name.lower() == 'BA_2Motifs'.lower():
            url = self.url.format(self.names[self.name][1])
            path = download_url(url, self.raw_dir)

    def process(self):
        """Generate synthetic graohs and convert them into Pytorch geometric Data object.

        Returns:
            Data: converted synthetic Pytorch geometric Data object
        """
        if self.name.lower() == 'BA_2Motifs'.lower():
            data_list = self.read_ba2motif_data(self.raw_dir, self.names[self.name][2])

            if self.pre_filter is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [data for data in data_list if self.pre_filter(data)]
                self.data, self.slices = self.collate(data_list)

            if self.pre_transform is not None:
                data_list = [self.get(idx) for idx in range(len(self))]
                data_list = [self.pre_transform(data) for data in data_list]
                self.data, self.slices = self.collate(data_list)
        
        else:
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

            data = data if self.pre_transform is None else self.pre_transform(data)
            data_list = [data]

        torch.save(self.collate(data_list), self.processed_paths[0])

    def gen_motif_edge_mask(self, data, node_idx=0, num_hops=3):
        if self.name in ['ba_2motifs']:
            return torch.logical_and(data.edge_index[0] >= 20, data.edge_index[1] >= 20)
        elif self.name in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            """ selection in a loop way to fetch all the nodes in the connected motifs """
            if data.y[node_idx] == 0:
                return torch.zeros_like(data.edge_index[0]).type(torch.bool)
            connected_motif_nodes = set()
            edge_label_matrix = data.edge_label_matrix + data.edge_label_matrix.T
            edge_index = data.edge_index.to('cpu')
            if isinstance(node_idx, torch.Tensor):
                connected_motif_nodes.add(node_idx.item())
            else:
                connected_motif_nodes.add(node_idx)
            for _ in range(num_hops):
                append_node = set()
                for node in connected_motif_nodes:
                    append_node.update(tuple(torch.where(edge_label_matrix[node] != 0)[0].tolist()))
                connected_motif_nodes.update(append_node)
            connected_motif_nodes_tensor = torch.Tensor(list(connected_motif_nodes))
            frm_mask = (edge_index[0].unsqueeze(1) - connected_motif_nodes_tensor.unsqueeze(0) == 0).any(dim=1)
            to_mask = (edge_index[1].unsqueeze(1) - connected_motif_nodes_tensor.unsqueeze(0) == 0).any(dim=1)
            return torch.logical_and(frm_mask, to_mask)

    def read_ba2motif_data(self, folder: str, prefix):
        with open(os.path.join(folder, f"{prefix}.pkl"), 'rb') as f:
            dense_edges, node_features, graph_labels = pickle.load(f)
        data_list = []
        adj_list = []
        max_num_nodes = 0
        for graph_idx in range(dense_edges.shape[0]):
            edge_index = dense_to_sparse(torch.from_numpy(dense_edges[graph_idx]))[0]
            data = Data(x=torch.ones((node_features[graph_idx].shape[0], 1),dtype=torch.float32),
                                edge_index=edge_index,
                                y=torch.from_numpy(np.where(graph_labels[graph_idx])[0]),
                                idx=graph_idx)
            data.edge_attr = torch.ones(data.edge_index.size(1))
            max_num_nodes = max(max_num_nodes, data.num_nodes)
            data.edge_mask = self.gen_motif_edge_mask(data)
            adj = from_edge_index_to_adj(data.edge_index, data.edge_attr, data.num_nodes)
            adj_list.append(adj)
            data_list.append(data)
        data_list = padded_datalist(data_list, adj_list, max_num_nodes)
        return data_list


    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))