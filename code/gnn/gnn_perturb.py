import math
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, GINEConv, TransformerConv
from torch.nn.parameter import Parameter
from code.utils.gen_utils import (
    convert_coo_to_tensor,
    from_adj_to_edge_index_torch,
    get_degree_matrix,
    create_symm_matrix_from_vec,
    create_vec_from_symm_matrix,
    from_edge_index_to_adj_torch,
)
from .model import GNN_basic, GNNPool
from torch_geometric.nn import GCNConv
import numpy as np


def get_existing_edge(new_edge_index, new_edge_weight, edge_index, edge_attr):
    keep_edge_idx = []
    for i in range(len(new_edge_index.T)):
        elmt = np.array(new_edge_index.T[i])
        pos_new_edge = np.where(np.all(np.array(edge_index.T)==elmt,axis=1))[0]
        if pos_new_edge.size > 0:
            keep_edge_idx.append(pos_new_edge[0])
    kept_edges = edge_index.T[keep_edge_idx]
    kept_edges = np.array(kept_edges)
    kept_edge_attr = edge_attr[keep_edge_idx]
    kept_edge_weight = new_edge_weight[keep_edge_idx]
    if kept_edges.ndim == 1:
        kept_edges = kept_edges.reshape(0,2)
    return(kept_edges, kept_edge_attr, kept_edge_weight)

def get_new_edge(new_edge_index, new_edge_weight, edge_index, edge_attr):
    new_added_edges = []
    new_added_edge_idx = []
    for i in range(len(new_edge_index.T)):
        elmt = np.array(new_edge_index.T[i])
        pos_new_edge = np.where(np.all(np.array(edge_index.T)==elmt,axis=1))[0]
        if pos_new_edge.size == 0:
            new_added_edges.append(elmt)
            new_added_edge_idx.append(i)
    new_added_edges = np.array(new_added_edges)
    mean_feat = np.mean(np.array(edge_attr),0)
    var_feat = np.var(np.array(edge_attr),0)
    new_added_edge_attr= np.array([np.random.normal(loc=mean_feat[i], scale=var_feat[i], size=(len(new_added_edges))) for i in range(edge_attr.shape[1])]).T
    new_added_edge_weight = new_edge_weight[new_added_edge_idx]
    if new_added_edges.ndim == 1:
        new_added_edges = new_added_edges.reshape(0,2)
    return(new_added_edges, new_added_edge_attr, new_added_edge_weight)


class GNNPerturb(GNN_basic):
    def __init__(
        self, input_dim, output_dim, adj, model_params, beta, edge_additions=False
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.num_nodes = self.adj.shape[0]
        self.edge_dim = model_params["edge_dim"]
        self.num_layers = model_params["num_layers"]
        self.hidden_dim = model_params["hidden_dim"]
        self.dropout = model_params["dropout"]
        # readout
        self.readout = model_params["readout"]
        self.readout_layer = GNNPool(self.readout)
        self.beta = beta
        self.edge_additions = (
            edge_additions  # are edge additions included in perturbed matrix
        )
        # P_hat needs to be symmetric ==> learn vector representing entries in upper/lower triangular matrix and use to populate P_hat later
        self.P_vec_size = (
            int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes
        )
        if self.edge_additions:
            self.P_vec = Parameter(torch.FloatTensor(torch.zeros(self.P_vec_size)))
        else:
            # self.P_vec = Parameter(torch.FloatTensor(torch.rand(self.P_vec_size)*(-0.5)+1))
            self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.reset_parameters()
        self.get_layers()
        
    def reset_parameters(self, eps=10**-4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(
                    self.P_vec, torch.FloatTensor(adj_vec)
                )  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

    def get_layers(self):
        # GNN layers
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        self.probs = F.softmax(self.logits, dim=1)
        return F.log_softmax(self.logits, dim=1)

    def forward_prediction(self, *args, **kwargs):
        _, _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb, self.P = self.get_emb_prediction(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        self.probs = F.softmax(self.logits, dim=1)
        return F.log_softmax(self.logits, dim=1), self.P

    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)
        self.sub_adj = from_edge_index_to_adj_torch(
            edge_index, edge_weight, self.num_nodes
        )
        # x, self.sub_adj = kwargs.get("x"), kwargs.get("adj")
        # Get adj matrix with only edges that have nonzero attention weights
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(
            self.P_vec, self.num_nodes
        )  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_additions:  # Learn new adj matrix directly
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(
                self.num_nodes
            )  # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(
                self.num_nodes
            )  # Use sigmoid to bound P_hat in [0,1]
        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        new_edge_index, new_edge_weight = from_adj_to_edge_index_torch(norm_adj)
        new_edge_weight = new_edge_weight.detach().numpy()
        kept_edges, kept_edge_attr, kept_edge_weight = get_existing_edge(new_edge_index, new_edge_weight, edge_index, edge_attr)
        new_added_edges, new_added_edge_attr, new_added_edge_weight = get_new_edge(new_edge_index, new_edge_weight, edge_index, edge_attr)
        new_edge_index = torch.LongTensor(np.concatenate((kept_edges, new_added_edges),0).T)
        new_edge_attr = torch.FloatTensor(np.concatenate((kept_edge_attr, new_added_edge_attr),0))
        new_edge_weight = torch.FloatTensor(np.concatenate((kept_edge_weight, new_added_edge_weight),0))

        for layer in self.convs:
            x = layer(x, new_edge_index, new_edge_attr*new_edge_weight[:,None])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def get_emb_prediction(self, *args, **kwargs):
        x, edge_index, edge_attr, edge_weight, _ = self._argsparse(*args, **kwargs)
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat
        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        new_edge_index, new_edge_weight = from_adj_to_edge_index_torch(norm_adj)
        new_edge_weight = new_edge_weight.detach().numpy()
        kept_edges, kept_edge_attr, kept_edge_weight = get_existing_edge(new_edge_index, new_edge_weight, edge_index, edge_attr)
        new_added_edges, new_added_edge_attr, new_added_edge_weight = get_new_edge(new_edge_index, new_edge_weight, edge_index, edge_attr)
        new_edge_index = torch.LongTensor(np.concatenate((kept_edges, np.array(new_added_edges)),0).T)
        new_edge_attr = torch.FloatTensor(np.concatenate((kept_edge_attr, new_added_edge_attr),0))
        new_edge_weight = torch.FloatTensor(np.concatenate((kept_edge_weight, new_added_edge_weight),0))
        ###################
        for layer in self.convs:
            x = layer(x, new_edge_index, new_edge_attr*new_edge_weight[:,None])
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x, self.P

    def loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        # Need dim >=2 for F.nll_loss to work
        if output.ndim < 2:
            output = output.unsqueeze(0)
            y_pred_orig = y_pred_orig.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P
        else:
            cf_adj = self.P * self.adj
        cf_adj.requires_grad = (
            True  # Need to change this otherwise loss_graph_dist has no gradient
        )
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        loss_pred = -F.nll_loss(output, y_pred_orig)
        loss_graph_dist = (
            sum(sum(abs(cf_adj - self.adj))) / 2
        )  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj

    def get_pred_label(self, pred):
        return pred.argmax(dim=1)


class GCNPerturb(GNNPerturb):
    def __init__(self, input_dim, output_dim, adj, model_params, beta, edge_additions=False):
        super().__init__(input_dim, output_dim, adj, model_params, beta, edge_additions=edge_additions)

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(GCNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return



class GATPerturb(GNNPerturb):
    def __init__(self, input_dim, output_dim, adj, model_params, beta, edge_additions=False):
        super().__init__(input_dim, output_dim, adj, model_params, beta, edge_additions=edge_additions)

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GATConv(current_dim, self.hidden_dim, edge_dim=self.edge_dim)
            )
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return

class GINPerturb(GNNPerturb):
    def __init__(self, input_dim, output_dim, adj, model_params, beta, edge_additions=False):
        super().__init__(input_dim, output_dim, adj, model_params, beta, edge_additions=edge_additions)

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(current_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    ),
                    edge_dim=self.edge_dim,
                )
            )
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return


class TRANSFORMERPerturb(GNNPerturb):
    def __init__(self, input_dim, output_dim, adj, model_params, beta, edge_additions=False):
        super().__init__(input_dim, output_dim, adj, model_params, beta, edge_additions=edge_additions)

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            if l == 0:
                self.convs.append(
                TransformerConv(current_dim, self.hidden_dim, heads=4, edge_dim=self.edge_dim)#, concat=False)
                )
            else:
                current_dim = self.hidden_dim * 4
                self.convs.append(
                TransformerConv(current_dim, current_dim, edge_dim=self.edge_dim)#, concat=False)
                )
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return
