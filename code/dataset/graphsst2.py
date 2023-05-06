
 

# From https://github.com/divelab/DIG/blob/main/dig/xgraph/datasets/load_datasets.py

import os
import yaml
import glob
import json
import random
import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from utils.gen_utils import padded_datalist, from_edge_index_to_adj

from pathlib import Path

def undirected_graph(data):
    data.edge_index = torch.cat([torch.stack([data.edge_index[1], data.edge_index[0]], dim=0),
                                 data.edge_index], dim=1)
    data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    return data


def split(data, batch):
    #print('batch', batch)
    #print('bincount batch', np.bincount(batch))
    # i-th contains elements from slice[i] to slice[i+1]
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])
    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    #print('node_slice', node_slice)
    #print('edge_slice', edge_slice)

    #print('edge_index before', data.edge_index)
    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = np.bincount(batch).tolist()

    #print('edge_index after', data.edge_index)

    slices = dict()
    slices['x'] = node_slice
    slices['edge_index'] = edge_slice
    slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    slices['sentence_tokens'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    return data, slices


def read_file(folder, prefix, name):
    file_path = osp.join(folder, prefix + f'_{name}.txt')
    return np.genfromtxt(file_path, dtype=np.int64)


def read_sentigraph_data(folder: str, prefix: str):
    txt_files = glob.glob(os.path.join(folder, "{}_*.txt".format(prefix)))
    json_files = glob.glob(os.path.join(folder, "{}_*.json".format(prefix)))
    txt_names = [f.split(os.sep)[-1][len(prefix) + 1:-4] for f in txt_files]
    json_names = [f.split(os.sep)[-1][len(prefix) + 1:-5] for f in json_files]
    names = txt_names + json_names

    with open(os.path.join(folder, prefix+"_node_features.pkl"), 'rb') as f:
        x: np.array = pickle.load(f)
    x: torch.FloatTensor = torch.from_numpy(x)
    edge_index: np.array = read_file(folder, prefix, 'edge_index')
    edge_index: torch.tensor = torch.tensor(edge_index, dtype=torch.long).T
    batch: np.array = read_file(folder, prefix, 'node_indicator') - 1     # from zero
    y: np.array = read_file(folder, prefix, 'graph_labels')
    y: torch.tensor = torch.tensor(y, dtype=torch.long)
    #edge_attr: torch.tensor = torch.ones((edge_index.size(1), 1), dtype=torch.float)
    
    name = torch.tensor(range(y.size(0)))
    supplement = dict()
    if 'split_indices' in names:
        split_indices: np.array = read_file(folder, prefix, 'split_indices')
        split_indices = torch.tensor(split_indices, dtype=torch.long)
        supplement['split_indices'] = split_indices
    if 'sentence_tokens' in names:
        with open(os.path.join(folder, prefix + '_sentence_tokens.json')) as f:
            sentence_tokens: dict = json.load(f)
        supplement['sentence_tokens'] = sentence_tokens
    data = Data(name=name, x=x, edge_index=edge_index, y=y.reshape(-1, 1).float(), sentence_tokens=list(sentence_tokens.values()))
    #data = Data(name=name, x=x, edge_index=edge_index, edge_attr=edge_attr, y=y.reshape(-1, 1).float(), sentence_tokens=list(sentence_tokens.values()))
    #print(data)
    data, slices = split(data, batch)

    return data, slices, supplement


class SentiGraphDataset(InMemoryDataset):
    names = ['graphsst2', 'Graph-SST2']
    def __init__(self, root, name, transform=None, pre_transform=undirected_graph):
        self.name = name.lower()
        assert self.name in self.names
        super(SentiGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.supplement = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['node_features', 'node_indicator', 'sentence_tokens', 'edge_index',
                'graph_labels', 'split_indices']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        self.data, self.slices, self.supplement \
              = read_sentigraph_data(self.raw_dir, self.names[1])
        
        data_list = []
        adj_list = []
        max_num_nodes = 0
        graph_idx = 0
        for idx in range(len(self)):
            data = self.get(idx)
            max_num_nodes = max(max_num_nodes, data.num_nodes)
            adj = from_edge_index_to_adj(data.edge_index, None, data.num_nodes)
            data.idx = graph_idx
            data.edge_attr = torch.ones((data.edge_index.size(1), 1), dtype=torch.float)
            adj_list.append(adj)
            data_list.append(data)
            graph_idx += 1
        data_list = padded_datalist(data_list, adj_list, max_num_nodes)
        self.data, self.slices = self.collate(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices, self.supplement), self.processed_paths[0])


class SentiGraphTransform(object):

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, data):
        data.edge_attr = torch.ones(data.edge_index.size(1), 1)
        # integrate further transform
        if self.transform is not None:
            return self.transform(data)
        return data

def load_SeniGraph(dataset_dir, dataset_name, transform=None):
    sent_transform = SentiGraphTransform(transform)
    dataset = SentiGraphDataset(root=dataset_dir, name=dataset_name, transform=sent_transform)
    return dataset
