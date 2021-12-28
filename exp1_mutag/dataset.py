import os
import os.path as osp
import zipfile
import gzip

import numpy as np

import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.data.collate import collate

from typing import Optional, List, Tuple, Dict

def collate_data(data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(os.path.basename(path).split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.
    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def process_mutag(raw_data_dir):
    with open(os.path.join(raw_data_dir, 'MUTAG_node_labels.txt'), 'r') as f:
        nodes_all_temp = f.read().splitlines()
        nodes_all = [int(i) for i in nodes_all_temp]

    adj_all = np.zeros((len(nodes_all), len(nodes_all)))
    with open(os.path.join(raw_data_dir, 'MUTAG_A.txt'), 'r') as f:
        adj_list = f.read().splitlines()
    for item in adj_list:
        lr = item.split(', ')
        l = int(lr[0])
        r = int(lr[1])
        adj_all[l - 1, r - 1] = 1

    with open(os.path.join(raw_data_dir, 'MUTAG_graph_indicator.txt'), 'r') as f:
        graph_indicator_temp = f.read().splitlines()
        graph_indicator = [int(i) for i in graph_indicator_temp]
        graph_indicator = np.array(graph_indicator)

    with open(os.path.join(raw_data_dir, 'MUTAG_graph_labels.txt'), 'r') as f:
        graph_labels_temp = f.read().splitlines()
        graph_labels = [int(i) for i in graph_labels_temp]

    data_list = []
    for i in range(1, 189):
        idx = np.where(graph_indicator == i)
        graph_len = len(idx[0])
        adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
        label = int(graph_labels[i - 1] == 1)
        feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
        nb_clss = 7
        targets = np.array(feature).reshape(-1)
        one_hot_feature = np.eye(nb_clss)[targets]
        data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                            edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                            y=label)
        data_list.append(data_example)
    return data_list