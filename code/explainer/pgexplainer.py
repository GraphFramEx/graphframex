"""
Description: The implement of PGExplainer model
<https://arxiv.org/abs/2011.04573>
"""

import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
import networkx as nx
from math import sqrt
from torch import Tensor
from textwrap import wrap
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple, List, Dict, Optional
from .shapley import gnn_score, GnnNetsNC2valueFunc, GnnNetsGC2valueFunc, sparsity
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem

EPS = 1e-6


def k_hop_subgraph_with_default_whole_graph(
    edge_index,
    node_idx=None,
    num_hops=3,
    relabel_nodes=False,
    num_nodes=None,
    flow="source_to_target",
):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ["source_to_target", "target_to_source"]
    if flow == "target_to_source":
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, np.int64, list, tuple)):
            node_idx = torch.tensor(
                [node_idx], device=row.device, dtype=torch.int64
            ).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[: node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return (
        subset,
        edge_index,
        inv,
        edge_mask,
    )  # subset: key new node idx; value original node idx


def calculate_selected_nodes(data, edge_mask, top_k):
    threshold = float(
        edge_mask.reshape(-1)
        .sort(descending=True)
        .values[min(top_k, edge_mask.shape[0] - 1)]
    )
    hard_mask = (edge_mask > threshold).cpu()
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    edge_index = data.edge_index.cpu().numpy()
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes


class PGExplainer(nn.Module):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.

    Args:
        model (:class:`torch.nn.Module`): The target model prepared to explain
        in_channels (:obj:`int`): Number of input channels for the explanation network
        explain_graph (:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        epochs (:obj:`int`): Number of epochs to train the explanation network
        lr (:obj:`float`): Learning rate to train the explanation network
        coff_size (:obj:`float`): Size regularization to constrain the explanation size
        coff_ent (:obj:`float`): Entropy regularization to constrain the connectivity of explanation
        t0 (:obj:`float`): The temperature at the first epoch
        t1(:obj:`float`): The temperature at the final epoch
        num_hops (:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
        (default: :obj:`None`)

    .. note: For node classification model, the :attr:`explain_graph` flag is False.
      If :attr:`num_hops` is set to :obj:`None`, it will be automatically calculated by calculating the
      :class:`torch_geometric.nn.MessagePassing` layers in the :attr:`model`.

    """

    def __init__(
        self,
        model,
        in_channels: int,
        device,
        explain_graph: bool = True,
        epochs: int = 20,
        lr: float = 0.005,
        coff_size: float = 0.01,
        coff_ent: float = 5e-4,
        t0: float = 5.0,
        t1: float = 1.0,
        sample_bias: float = 0.0,
        num_hops: Optional[int] = None,
    ):
        super(PGExplainer, self).__init__()
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.in_channels = in_channels
        self.explain_graph = explain_graph

        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1
        self.sample_bias = sample_bias

        self.num_hops = self.update_num_hops(num_hops)
        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        r"""Set the edge weights before message passing

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)

        The :attr:`edge_mask` will be randomly initialized when set to :obj:`None`.

        .. note:: When you use the :meth:`~PGExplainer.__set_masks__`,
          the explain flag for all the :class:`torch_geometric.nn.MessagePassing`
          modules in :attr:`model` will be assigned with :obj:`True`. In addition,
          the :attr:`edge_mask` will be assigned to all the modules.
          Please take :meth:`~PGExplainer.__clear_masks__` to reset.
        """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.randn(E) * std + init_bias
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """clear the edge weights to None, and set the explain flag to :obj:`False`"""
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def update_num_hops(self, num_hops: int):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return "source_to_target"

    def __loss__(self, prob: Tensor, ori_pred: int):
        logit = prob[ori_pred]
        logit = logit + EPS
        pred_loss = -torch.log(logit)

        # size
        edge_mask = self.sparse_mask_values
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy
        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = -edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(
            1 - edge_mask
        )
        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def get_subgraph(
        self,
        node_idx: int,
        x: Tensor,
        edge_index: Tensor,
        y: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, List, Dict]:
        r"""extract the subgraph of target node

        Args:
            node_idx (:obj:`int`): The node index
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            y (:obj:`torch.Tensor`, :obj:`None`): Node label matrix with shape :obj:`[num_nodes]`
              (default :obj:`None`)
            kwargs(:obj:`Dict`, :obj:`None`): Additional parameters

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`,
          :obj:`List`, :class:`Dict`)

        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index,
            node_idx,
            self.num_hops,
            relabel_nodes=True,
            num_nodes=num_nodes,
            flow=self.__flow__(),
        )

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        if y is not None:
            y = y[subset]
        return x, edge_index, y, subset, edge_mask, kwargs

    def concrete_sample(
        self, log_alpha: Tensor, beta: float = 1.0, training: bool = True
    ):
        r"""Sample from the instantiation of concrete distribution when training"""
        if training:
            bias = self.sample_bias
            random_noise = torch.rand(log_alpha.shape) * (1 - 2 * bias) + bias
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def explain_node(self, node_idx, x, edge_index):
        select_edge_index = torch.arange(0, edge_index.shape[1])
        (
            subgraph_x,
            subgraph_edge_index,
            _,
            subset,
            subgraph_edge_mask,
            kwargs,
        ) = self.get_subgraph(
            node_idx, x, edge_index, select_edge_index=select_edge_index
        )
        self.select_edge_mask = edge_index.new_empty(
            edge_index.size(1), device=self.device, dtype=torch.bool
        )
        self.select_edge_mask.fill_(False)
        self.select_edge_mask[select_edge_index] = True
        self.hard_edge_mask = edge_index.new_empty(
            subgraph_edge_index.size(1), device=self.device, dtype=torch.bool
        )
        self.hard_edge_mask.fill_(True)
        self.subset = subset
        self.new_node_idx = torch.where(subset == node_idx)[0]

        subgraph_embed = self.model.get_emb(subgraph_x, subgraph_edge_index)
        _, edge_mask = self.explain(
            subgraph_x,
            subgraph_edge_index,
            embed=subgraph_embed,
            tmp=1.0,
            training=False,
            node_idx=self.new_node_idx,
        )
        subgraph_edge_mask = subgraph_edge_mask.cpu().detach().numpy()
        subindices = np.where(subgraph_edge_mask > 0)[0]
        edge_mask_full = torch.zeros(len(subgraph_edge_mask))
        for i in range(len(subindices)):
            edge_mask_full[subindices[i]] = edge_mask[i]
        return edge_mask_full

    def explain(
        self,
        x: Tensor,
        edge_index: Tensor,
        embed: Tensor,
        tmp: float = 1.0,
        training: bool = False,
        **kwargs,
    ) -> Tuple[float, Tensor]:
        r"""explain the GNN behavior for graph with explanation network

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            embed (:obj:`torch.Tensor`): Node embedding matrix with shape :obj:`[num_nodes, dim_embedding]`
            tmp (:obj`float`): The temperature parameter fed to the sample procedure
            training (:obj:`bool`): Whether in training procedure or not

        Returns:
            probs (:obj:`torch.Tensor`): The classification probability for graph with edge mask
            edge_mask (:obj:`torch.Tensor`): The probability mask for graph edges
        """
        node_idx = kwargs.get("node_idx")
        nodesize = embed.shape[0]
        if self.explain_graph:
            col, row = edge_index
            f1 = embed[col]
            f2 = embed[row]
            f12self = torch.cat([f1, f2], dim=-1)
        else:
            col, row = edge_index
            f1 = embed[col]
            f2 = embed[row]
            self_embed = embed[node_idx].repeat(f1.shape[0], 1)
            f12self = torch.cat([f1, f2, self_embed], dim=-1)

        # using the node embedding to calculate the edge weight
        h = f12self.to(self.device)
        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        values = self.concrete_sample(values, beta=tmp, training=training)
        self.sparse_mask_values = values
        mask_sparse = torch.sparse_coo_tensor(edge_index, values, (nodesize, nodesize))
        mask_sigmoid = mask_sparse.to_dense()
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        logits = self.model(x, edge_index)
        probs = F.softmax(logits, dim=-1)

        self.__clear_masks__()
        return probs, edge_mask

    def train_explanation_network(self, dataset):
        r"""training the explanation network by gradient descent(GD) using Adam optimizer"""
        optimizer = Adam(self.elayers.parameters(), lr=self.lr)
        if self.explain_graph:
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(self.device)
                    logits = self.model(data.x, data.edge_index)
                    emb = self.model.get_emb(data.x, data.edge_index)
                    emb_dict[gid] = emb.data.cpu()
                    ori_pred_dict[gid] = logits.argmax(-1).data.cpu()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                pred_list = []
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    data.to(self.device)
                    prob, edge_mask = self.explain(
                        data.x,
                        data.edge_index,
                        embed=emb_dict[gid],
                        tmp=tmp,
                        training=True,
                    )
                    loss_tmp = self.__loss__(prob.squeeze(), ori_pred_dict[gid])
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob.argmax(-1).item()
                    pred_list.append(pred_label)

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f"Epoch: {epoch} | Loss: {loss}")
        else:
            with torch.no_grad():
                data = dataset  # [0]
                data.to(self.device)
                self.model.eval()
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                pred_dict = {}
                logits = self.model(data.x, data.edge_index)
                for node_idx in tqdm.tqdm(explain_node_index_list):
                    pred_dict[node_idx] = logits[node_idx].argmax(-1).item()

            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                optimizer.zero_grad()
                tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                self.elayers.train()
                tic = time.perf_counter()
                for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                    with torch.no_grad():
                        x, edge_index, y, subset, _ = self.get_subgraph(
                            node_idx=node_idx,
                            x=data.x,
                            edge_index=data.edge_index,
                            y=data.y,
                        )
                        emb = self.model.get_emb(x, edge_index)
                        new_node_index = int(torch.where(subset == node_idx)[0])
                    pred, edge_mask = self.explain(
                        x, edge_index, emb, tmp, training=True, node_idx=new_node_index
                    )
                    loss_tmp = self.__loss__(pred[new_node_index], pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f"Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}")
            print(f"training time is {duration:.5}s")

    def __repr__(self):
        return f"{self.__class__.__name__}()"
