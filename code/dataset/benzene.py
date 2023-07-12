import os
import torch
import numpy as np
import os.path as osp
import pandas as pd
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from utils.gen_utils import padded_datalist, from_edge_index_to_adj


ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "Na", "Ca", "I", "B", "H", "*"]


def edge_mask_from_node_mask(node_mask: torch.Tensor, edge_index: torch.Tensor):
    """
    Convert edge_mask to node_mask

    Args:
        node_mask (torch.Tensor): Boolean mask over all nodes included in edge_index. Indices must
            match to those in edge index. This is straightforward for graph-level prediction, but
            converting over subgraphs must be done carefully to match indices in both edge_index and
            the node_mask.
    """

    node_numbers = node_mask.nonzero(as_tuple=True)[0]

    iter_mask = torch.zeros((edge_index.shape[1],))

    # See if edges have both ends in the node mask
    for i in range(edge_index.shape[1]):
        iter_mask[i] = (edge_index[0, i] in node_numbers) and (
            edge_index[1, i] in node_numbers
        )

    return iter_mask


class Benzene(InMemoryDataset):
    def __init__(self, root, name):
        self.name = name.lower()
        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return ["data.pt"]

    # def download(self):
    # raise NotImplementedError

    def process(self):
        data = np.load(self.raw_dir + "/benzene.npz", allow_pickle=True)

        data_list = []
        adj_list = []
        max_num_nodes = 0
        mol_idx = 0

        att, X, y, df = data["attr"], data["X"], data["y"], data["smiles"]
        ylist = [y[i][0] for i in range(y.shape[0])]
        X = X[0]

        for i in range(len(X)):
            x = torch.from_numpy(X[i]["nodes"])
            edge_attr = torch.from_numpy(X[i]["edges"])
            # y = X[i]['globals'][0]
            y = torch.tensor([ylist[i]], dtype=torch.long)

            # Get edge_index:
            e1 = torch.from_numpy(X[i]["receivers"]).long()
            e2 = torch.from_numpy(X[i]["senders"]).long()

            edge_index = torch.stack([e1, e2])

            # Get ground-truth explanation:
            node_imp = torch.from_numpy(att[i][0]["nodes"]).float()

            # Error-check:
            assert (
                att[i][0]["n_edge"] == X[i]["n_edge"]
            ), "Num: {}, Edges different sizes".format(i)
            assert node_imp.shape[0] == x.shape[0], "Num: {}, Shapes: {} vs. {}".format(
                i, node_imp.shape[0], x.shape[0]
            ) + "\nExp: {} \nReal:{}".format(att[i][0], X[i])

            i_exps = []

            for j in range(node_imp.shape[1]):
                edge_imp = edge_mask_from_node_mask(
                    node_imp[:, j].bool(), edge_index=edge_index
                )
                i_exps.append(edge_imp)

            gt_edge_mask = torch.max(
                torch.stack([edge_imp for edge_imp in i_exps]), dim=0
            )[0]

            data_i = Data(
                x=x,
                y=y,
                edge_attr=edge_attr,
                edge_index=edge_index,
                edge_mask=gt_edge_mask,
                idx=mol_idx,
            )

            adj = from_edge_index_to_adj(data_i.edge_index, None, data_i.num_nodes)
            adj_list.append(adj)

            if self.pre_filter is not None and not self.pre_filter(data_i):
                continue

            if self.pre_transform is not None:
                data_i = self.pre_transform(data_i)

            max_num_nodes = max(max_num_nodes, data_i.num_nodes)
            data_list.append(data_i)
            mol_idx += 1

        data_list = padded_datalist(data_list, adj_list, max_num_nodes)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
