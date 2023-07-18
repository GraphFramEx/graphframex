# adapt from https://github.com/bknyaz/graph_attention_pool/blob/master/graphdata.py

import numpy as np
import os.path as osp
import pickle
import torch
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data

import yaml
from pathlib import Path
from torch_geometric.loader import DataLoader
from utils.gen_utils import padded_datalist, from_edge_index_to_adj


def compute_adjacency_matrix_images(coord, sigma=0.1):
    coord = coord.reshape(-1, 2)
    dist = cdist(coord, coord)
    A = np.exp(-dist / (sigma * np.pi) ** 2)
    A[np.diag_indices_from(A)] = 0
    return A


def list_to_torch(data):
    for i in range(len(data)):
        if data[i] is None:
            continue
        elif isinstance(data[i], np.ndarray):
            if data[i].dtype == np.bool:
                data[i] = data[i].astype(np.float32)
            data[i] = torch.from_numpy(data[i]).float()
        elif isinstance(data[i], list):
            data[i] = list_to_torch(data[i])
    return data


class MNIST75sp_Binary(InMemoryDataset):
    splits = ["test", "train"]

    def __init__(
        self,
        root,
        name,
        use_mean_px=True,
        use_coord=True,
        node_gt_att_threshold=0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.node_gt_att_threshold = node_gt_att_threshold
        self.use_mean_px, self.use_coord = use_mean_px, use_coord
        self.name = name.lower()
        super(MNIST75sp_Binary, self).__init__(
            root, transform, pre_transform, pre_filter
        )
        # idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return ["mnist_75sp_train.pkl", "mnist_75sp_test.pkl"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        for file in self.raw_file_names:
            if not osp.exists(osp.join(self.raw_dir, file)):
                print(
                    "raw data of `{}` doesn't exist, please download from our github.".format(
                        file
                    )
                )
                raise FileNotFoundError

    def process(self):

        data_list = []
        adj_list = []
        max_num_nodes = 0
        img_idx = 0

        for mode in self.splits:

            data_file = "mnist_75sp_%s.pkl" % mode
            with open(osp.join(self.raw_dir, data_file), "rb") as f:
                self.labels, self.sp_data = pickle.load(f)

            sp_file = "mnist_75sp_%s_superpixels.pkl" % mode
            with open(osp.join(self.raw_dir, sp_file), "rb") as f:
                self.all_superpixels = pickle.load(f)

            self.use_mean_px = self.use_mean_px
            self.use_coord = self.use_coord
            self.n_samples = len(self.labels)
            self.img_size = 28
            self.node_gt_att_threshold = self.node_gt_att_threshold

            (
                self.edge_indices,
                self.xs,
                self.edge_attrs,
                self.node_gt_atts,
                self.edge_gt_atts,
            ) = ([], [], [], [], [])

            for index, sample in enumerate(self.sp_data):

                data_y = torch.LongTensor([self.labels[index]])
                if data_y.item() == 0 or data_y.item() == 1:
                    mean_px, coord, sp_order = sample[:3]
                    superpixels = self.all_superpixels[index]
                    coord = coord / self.img_size
                    A = compute_adjacency_matrix_images(coord)
                    N_nodes = A.shape[0]

                    A = torch.FloatTensor((A > 0.1) * A)
                    edge_index, edge_attr = dense_to_sparse(A)

                    x = None
                    if self.use_mean_px:
                        x = mean_px.reshape(N_nodes, -1)
                    if self.use_coord:
                        coord = coord.reshape(N_nodes, 2)
                        if self.use_mean_px:
                            x = np.concatenate((x, coord), axis=1)
                        else:
                            x = coord
                    if x is None:
                        x = np.ones(N_nodes, 1)  # dummy features

                    # replicate features to make it possible to test on colored images
                    x = np.pad(x, ((0, 0), (2, 0)), "edge")
                    if self.node_gt_att_threshold == 0:
                        node_gt_att = (mean_px > 0).astype(np.float32)
                    else:
                        node_gt_att = mean_px.copy()
                        node_gt_att[node_gt_att < self.node_gt_att_threshold] = 0

                    node_gt_att = torch.LongTensor(node_gt_att).view(-1)
                    row, col = edge_index
                    edge_gt_att = torch.LongTensor(
                        node_gt_att[row] * node_gt_att[col]
                    ).view(-1)

                    data = Data(
                        x=torch.tensor(x),
                        y=data_y,
                        edge_index=edge_index,
                        edge_attr=edge_attr.reshape(-1, 1),
                        node_label=node_gt_att.float(),
                        edge_mask=edge_gt_att.float(),
                        sp_order=torch.tensor(sp_order),
                        superpixels=torch.tensor(superpixels),
                        name=f"MNISTSP-{mode}-{index}",
                        idx=img_idx,
                    )
                    max_num_nodes = max(max_num_nodes, data.num_nodes)
                    adj = from_edge_index_to_adj(data.edge_index, None, data.num_nodes)

                    adj_list.append(adj)
                    data_list.append(data)
                    img_idx += 1

                    # idx = self.processed_file_names.index('mnist_75sp_{}.pt'.format(mode))
                    # torch.save(self.collate(data_list), self.processed_paths[idx])

        data_list = padded_datalist(data_list, adj_list, max_num_nodes)
        data, slices = self.collate(data_list)
        print("Number of graphs with label 0: ", (data.y == 0).sum().item())
        print("Number of graphs with label 1: ", (data.y == 1).sum().item())

        torch.save((data, slices), self.processed_paths[0])
