import os.path as osp
import torch
import mat73
from sklearn.model_selection import train_test_split
import os
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import torch
from torch_geometric.data import Data
import numpy as np

"These classes produce power grids datasets that have the contingency included in the edge_index matrix and the edge_feature matrix, where at the contigency index the edge feature are zeros. The explanaibility max is the contingency"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_edgeorder(edge_order):
    return torch.tensor(edge_order["bList"]-1)


class UKCont(InMemoryDataset):
    # Base folder to download the files
    names = {
        "uk": ["uk", "uk", None, None]
    }
    raw_path = 'uk/'
    def __init__(self, root, name, datatype='multiclass', transform=None, pre_transform=None):

        self.datatype = datatype
        self.name = name.lower()
        assert self.name in self.names.keys()
        super(UKCont, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")


    @property
    def raw_file_names(self):
        # List of the raw files
        return ['Bf.mat',
                'blist.mat',
                'Ef.mat',
                'exp.mat',
                'of_bi.mat',
                'of_mc.mat',
                'of_reg.mat']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):
        
        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, 'blist.mat')
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, 'of_bi.mat')
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, 'of_reg.mat')
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, 'of_mc.mat')
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, 'Bf.mat')
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, 'Ef.mat')
        edge_f = mat73.loadmat(path)

        path = os.path.join(self.raw_dir, 'exp.mat')
        #exp = mat73.loadmat(path)

        node_f = node_f['B_f_tot']
        edge_f = edge_f['E_f_post']
        of_bi = of_bi['output_features']
        of_mc = of_mc['category']
        #exp_mask = exp['explainations']

        data_list = []

        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape([-1, 3]).to(device)
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask[cont] = 1
            e_mask_post = torch.cat((e_mask, e_mask), 0).to(device)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f.reshape([-1, 4]).type(torch.float32), f.reshape([-1, 4]).type(torch.float32)), 0).to(device)
            # flip branch list
            edge_iwr = torch.fliplr(edge_order.reshape(-1, 2).type(torch.long))
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_order.reshape(-1, 2).type(torch.long), edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            data_type = self.datatype
            if data_type == 'Binary' or data_type == 'binary':
                ydata = torch.tensor(of_bi[i][0], dtype=torch.float, device=device).view(1, -1)
            if data_type == 'Regression' or data_type == 'regression':
                ydata = torch.tensor(of_reg[i][0], dtype=torch.int, device=device).view(1, -1)
            if data_type == 'Multiclass' or data_type == 'multiclass':
                #do argmax
                ydata = torch.tensor(np.argmax(of_mc[i][0]), dtype=torch.int, device=device).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(x=x, edge_index=edge_iw, edge_attr=f_totw, y=ydata, edge_mask=e_mask_post)
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class IEEE24Cont(InMemoryDataset):
    # Base folder to download the files
    names = {
        "ieee24": ["ieee24", "ieee24", None, None]
    }
    raw_path = 'ieee24/'
    def __init__(self, root, name, datatype='Multiclass', transform=None, pre_transform=None):

        self.datatype = datatype
        self.name = name.lower()
        assert self.name in self.names.keys()
        super(IEEE24Cont, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['Bf.mat',
                'blist.mat',
                'Ef.mat',
                'exp.mat',
                'of_bi.mat',
                'of_mc.mat',
                'of_reg.mat']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):

        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, 'blist.mat')
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, 'of_bi.mat')
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, 'of_reg.mat')
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, 'of_mc.mat')
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, 'Bf.mat')
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, 'Ef.mat')
        edge_f = mat73.loadmat(path)

        path = os.path.join(self.raw_dir, 'exp.mat')


        node_f = node_f['B_f_tot']
        edge_f = edge_f['E_f_post']
        of_bi = of_bi['output_features']
        of_mc = of_mc['category']


        data_list = []

        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape([-1, 3]).to(device)
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask[cont] = 1
            e_mask_post = torch.cat((e_mask, e_mask), 0).to(device)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f.reshape([-1, 4]).type(torch.float32), f.reshape([-1, 4]).type(torch.float32)), 0).to(
                device)
            # flip branch list
            edge_iwr = torch.fliplr(edge_order.reshape(-1, 2).type(torch.long))
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_order.reshape(-1, 2).type(torch.long), edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            data_type = self.datatype
            if data_type == 'Binary' or data_type == 'binary':
                ydata = torch.tensor(of_bi[i][0], dtype=torch.float, device=device).view(1, -1)
            if data_type == 'Regression' or data_type == 'regression':
                ydata = torch.tensor(of_reg[i][0], dtype=torch.int, device=device).view(1, -1)
            if data_type == 'Multiclass' or data_type == 'multiclass':
                # do argmax
                ydata = torch.tensor(np.argmax(of_mc[i][0]), dtype=torch.int, device=device).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(x=x, edge_index=edge_iw, edge_attr=f_totw, y=ydata, edge_mask=e_mask_post)
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class IEEE39Cont(InMemoryDataset):
    # Base folder to download the files
    names = {
        "ieee39": ["ieee39", "ieee39", None, None]
    }
    raw_path = 'ieee39/'
    def __init__(self, root, name, datatype='multiclass', transform=None, pre_transform=None):

        self.datatype = datatype
        self.name = name.lower()
        assert self.name in self.names.keys()
        super(IEEE39Cont, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['Bf.mat',
                'blist.mat',
                'Ef.mat',
                'exp.mat',
                'of_bi.mat',
                'of_mc.mat',
                'of_reg.mat']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):

        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, 'blist.mat')
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, 'of_bi.mat')
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, 'of_reg.mat')
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, 'of_mc.mat')
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, 'Bf.mat')
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, 'Ef.mat')
        edge_f = mat73.loadmat(path)

        path = os.path.join(self.raw_dir, 'exp.mat')


        node_f = node_f['B_f_tot']
        edge_f = edge_f['E_f_post']
        of_bi = of_bi['output_features']
        of_mc = of_mc['category']


        data_list = []

        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = torch.tensor(node_f[i][0], dtype=torch.float32).reshape([-1, 3]).to(device)
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask[cont] = 1
            e_mask_post = torch.cat((e_mask, e_mask), 0).to(device)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f.reshape([-1, 4]).type(torch.float32), f.reshape([-1, 4]).type(torch.float32)), 0).to(
                device)
            # flip branch list
            edge_iwr = torch.fliplr(edge_order.reshape(-1, 2).type(torch.long))
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_order.reshape(-1, 2).type(torch.long), edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            data_type = self.datatype
            if data_type == 'Binary' or data_type == 'binary':
                ydata = torch.tensor(of_bi[i][0], dtype=torch.float, device=device).view(1, -1)
            if data_type == 'Regression' or data_type == 'regression':
                ydata = torch.tensor(of_reg[i][0], dtype=torch.int, device=device).view(1, -1)
            if data_type == 'Multiclass' or data_type == 'multiclass':
                # do argmax
                ydata = torch.tensor(np.argmax(of_mc[i][0]), dtype=torch.int, device=device).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(x=x, edge_index=edge_iw, edge_attr=f_totw, y=ydata, edge_mask=e_mask_post)
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])





