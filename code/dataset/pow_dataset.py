import os.path as osp
import torch
import mat73
from sklearn.model_selection import train_test_split
import os
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import torch
from torch_geometric.data import Data
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def index_edgeorder(edge_order):
    return torch.tensor(edge_order["bList"] - 1)


class UK(InMemoryDataset):
    # Base folder to download the files
    names = {"uk": ["uk", "uk", None, None]}
    raw_path = "uk+expmask/"
    url = "https://figshare.com/s/1af1dcbb4d7fc27b94e1"

    def __init__(
        self, root, name, datatype="Binary", transform=None, pre_transform=None
    ):

        self.datatype = datatype
        self.name = name.lower()
        super(UK, self).__init__(root, transform, pre_transform)
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
        return [
            "Bf.mat",
            "blist.mat",
            "Ef.mat",
            "exp.mat",
            "of_bi.mat",
            "of_mc.mat",
            "of_reg.mat",
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):
        # function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, "blist.mat")
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, "of_bi.mat")
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, "of_reg.mat")
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, "of_mc.mat")
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, "Bf.mat")
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, "Ef.mat")
        edge_f = mat73.loadmat(path)

        path = os.path.join(self.raw_dir, "exp.mat")
        exp = mat73.loadmat(path)

        node_f = node_f["B_f_tot"]
        edge_f = edge_f["E_f_post"]
        of_bi = of_bi["output_features"]
        of_mc = of_mc["category"]
        exp_mask = exp["explainations"]

        data_list = []

        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = (
                torch.tensor(node_f[i][0], dtype=torch.float32)
                .reshape([-1, 3])
                .to(device)
            )
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            if exp_mask[i][0] is None:  # .all() == 0:
                e_mask = e_mask
            else:
                e_mask[exp_mask[i][0].astype('int')-1] = 1
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask_post = th_delete(e_mask, cont)
            e_mask_post = torch.cat((e_mask_post, e_mask_post), 0).to(device)
            # remove edge features of the associated line
            f_tot = th_delete(f, cont).reshape([-1, 4]).type(torch.float32)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f_tot, f_tot), 0).to(device)
            # remove failed lines from branch list
            edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.long)
            # flip branch list
            edge_iwr = torch.fliplr(edge_iw)
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_iw, edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            data_type = self.datatype
            if data_type == "Binary" or data_type == "binary":
                ydata = torch.tensor(
                    of_bi[i][0], dtype=torch.float, device=device
                ).view(1, -1)
            if data_type == "Regression" or data_type == "regression":
                ydata = torch.tensor(of_reg[i][0], dtype=torch.int, device=device).view(
                    1, -1
                )
            if data_type == "Multiclass" or data_type == "multiclass":
                # do argmax
                ydata = torch.tensor(
                    np.argmax(of_mc[i][0]), dtype=torch.int, device=device
                ).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(
                x=x,
                edge_index=edge_iw,
                edge_attr=f_totw,
                y=ydata,
                edge_mask=e_mask_post,
            )
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class IEEE24(InMemoryDataset):
    # Base folder to download the files
    names = {"ieee24": ["ieee24", "ieee24", None, None]}
    raw_path = "ieee24+expmask/"

    def __init__(
        self, root, name, datatype="Binary", transform=None, pre_transform=None
    ):

        self.datatype = datatype
        self.name = name.lower()
        super(IEEE24, self).__init__(root, transform, pre_transform)
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
        return [
            "Bf.mat",
            "blist.mat",
            "Ef.mat",
            "exp.mat",
            "of_bi.mat",
            "of_mc.mat",
            "of_reg.mat",
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):
        # function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        raw_path = "ieee24+expmask/"

        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, "blist.mat")
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, "of_bi.mat")
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, "of_reg.mat")
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, "of_mc.mat")
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, "Bf.mat")
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, "Ef.mat")
        edge_f = mat73.loadmat(path)

        path = os.path.join(self.raw_dir, "exp.mat")
        exp = mat73.loadmat(path)

        node_f = node_f["B_f_tot"]
        edge_f = edge_f["E_f_post"]
        of_bi = of_bi["output_features"]
        of_mc = of_mc["category"]
        exp_mask = exp["explainations"]

        data_list = []

        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = (
                torch.tensor(node_f[i][0], dtype=torch.float32)
                .reshape([-1, 3])
                .to(device)
            )
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            if exp_mask[i][0] is None:  # .all() == 0:
                e_mask = e_mask
            else:
                e_mask[exp_mask[i][0].astype('int')-1] = 1
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask_post = th_delete(e_mask, cont)
            e_mask_post = torch.cat((e_mask_post, e_mask_post), 0).to(device)
            # remove edge features of the associated line
            f_tot = th_delete(f, cont).reshape([-1, 4]).type(torch.float32)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f_tot, f_tot), 0).to(device)
            # remove failed lines from branch list
            edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.long)
            # flip branch list
            edge_iwr = torch.fliplr(edge_iw)
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_iw, edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            data_type = self.datatype
            if data_type == "Binary" or data_type == "binary":
                ydata = torch.tensor(
                    of_bi[i][0], dtype=torch.float, device=device
                ).view(1, -1)
            if data_type == "Regression" or data_type == "regression":
                ydata = torch.tensor(of_reg[i][0], dtype=torch.int, device=device).view(
                    1, -1
                )
            if data_type == "Multiclass" or data_type == "multiclass":
                # do argmax
                ydata = torch.tensor(
                    np.argmax(of_mc[i][0]), dtype=torch.int, device=device
                ).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(
                x=x,
                edge_index=edge_iw,
                edge_attr=f_totw,
                y=ydata,
                edge_mask=e_mask_post,
            )
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class IEEE39(InMemoryDataset):
    # Base folder to download the files
    names = {"ieee39": ["ieee39", "ieee39", None, None]}
    raw_path = "ieee39+expmask/"

    def __init__(
        self, root, name, datatype="binary", transform=None, pre_transform=None
    ):

        self.datatype = datatype
        self.name = name.lower()
        super(IEEE39, self).__init__(root, transform, pre_transform)
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
        return [
            "Bf.mat",
            "blist.mat",
            "Ef.mat",
            "exp.mat",
            "of_bi.mat",
            "of_mc.mat",
            "of_reg.mat",
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        # Download the file specified in self.url and store
        # it in self.raw_dir
        path = self.raw_path

    def process(self):
        # function that deletes row
        def th_delete(tensor, indices):
            mask = torch.ones(tensor.size(), dtype=torch.bool)
            mask[indices] = False
            return tensor[mask]

        raw_path = "ieee39+expmask/"

        # load branch list also called edge order or edge index
        path = os.path.join(self.raw_dir, "blist.mat")
        edge_order = mat73.loadmat(path)
        edge_order = torch.tensor(edge_order["bList"] - 1)
        # load output binary classification labels
        path = os.path.join(self.raw_dir, "of_bi.mat")
        of_bi = mat73.loadmat(path)
        # load output binary regression labels
        path = os.path.join(self.raw_dir, "of_reg.mat")
        of_reg = mat73.loadmat(path)
        # load output mc labels
        path = os.path.join(self.raw_dir, "of_mc.mat")
        of_mc = mat73.loadmat(path)
        # load output node feature matrix
        path = os.path.join(self.raw_dir, "Bf.mat")
        node_f = mat73.loadmat(path)
        # load output edge feature matrix
        path = os.path.join(self.raw_dir, "Ef.mat")
        edge_f = mat73.loadmat(path)

        path = os.path.join(self.raw_dir, "exp.mat")
        exp = mat73.loadmat(path)

        node_f = node_f["B_f_tot"]
        edge_f = edge_f["E_f_post"]
        of_bi = of_bi["output_features"]
        of_mc = of_mc["category"]
        exp_mask = exp["explainations"]

        data_list = []

        # MAIN data processing loop
        for i in range(len(node_f)):
            # node feat
            x = (
                torch.tensor(node_f[i][0], dtype=torch.float32)
                .reshape([-1, 3])
                .to(device)
            )
            # edge feat
            f = torch.tensor(edge_f[i][0], dtype=torch.float32)
            e_mask = torch.zeros(len(edge_f[i][0]), 1)
            if exp_mask[i][0] is None:  # .all() == 0:
                e_mask = e_mask
            else:
                e_mask[exp_mask[i][0].astype('int')-1] = 1
            # contigency lists, finds where do we have contigencies from the .mat edge feature matrices
            # ( if a line is part of the contigency list all egde features are set 0)
            cont = [j for j in range(len(f)) if np.all(np.array(f[j])) == 0]
            e_mask_post = th_delete(e_mask, cont)
            e_mask_post = torch.cat((e_mask_post, e_mask_post), 0).to(device)
            # remove edge features of the associated line
            f_tot = th_delete(f, cont).reshape([-1, 4]).type(torch.float32)
            # concat the post-contigency edge feature matrix to take into account the reversed edges
            f_totw = torch.cat((f_tot, f_tot), 0).to(device)
            # remove failed lines from branch list
            edge_iw = th_delete(edge_order, cont).reshape(-1, 2).type(torch.long)
            # flip branch list
            edge_iwr = torch.fliplr(edge_iw)
            #  and concat the non flipped and flipped branch list
            edge_iw = torch.cat((edge_iw, edge_iwr), 0)
            edge_iw = edge_iw.t().contiguous().to(device)

            data_type = self.datatype
            if data_type == "Binary" or data_type == "binary":
                ydata = torch.tensor(
                    of_bi[i][0], dtype=torch.float, device=device
                ).view(1, -1)
            if data_type == "Regression" or data_type == "regression":
                ydata = torch.tensor(of_reg[i][0], dtype=torch.int, device=device).view(
                    1, -1
                )
            if data_type == "Multiclass" or data_type == "multiclass":
                # do argmax
                ydata = torch.tensor(
                    np.argmax(of_mc[i][0]), dtype=torch.int, device=device
                ).view(1, -1)

            # Fill Data object, 1 Data object -> 1 graph
            data = Data(
                x=x,
                edge_index=edge_iw,
                edge_attr=f_totw,
                y=ydata,
                edge_mask=e_mask_post,
            )
            # append Data object to datalist
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
