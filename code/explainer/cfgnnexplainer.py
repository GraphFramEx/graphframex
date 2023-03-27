# Based on https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
from utils.gen_utils import from_adj_to_edge_index_torch, get_degree_matrix
from gnn.gnn_perturb import GCNPerturb, GATPerturb, TRANSFORMERPerturb, GINPerturb
from torch_geometric.data import Data


class CFExplainer:
    """
    CF Explainer class, returns counterfactual subgraph
    """

    def __init__(
        self,
        model,
        cf_model_name,
        adj,
        feat,
        edge_feat,
        n_hid,
        dropout,
        readout,
        edge_dim,
        num_layers,
        labels,
        y_pred_orig,
        num_classes,
        beta,
        device,
    ):
        super(CFExplainer, self).__init__()
        self.model = model
        self.model.eval()
        self.cf_model_name = cf_model_name,
        self.adj = adj
        self.feat = feat
        self.edge_feat = edge_feat
        self.labels = labels
        self.y_pred_orig = y_pred_orig
        self.beta = beta
        self.num_classes = num_classes
        self.device = device
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.n_hid = n_hid
        self.dropout = dropout
        self.readout = readout

        cf_model_params = dict(
            edge_dim=edge_dim,
            num_layers=num_layers,
            hidden_dim=n_hid,
            dropout=dropout,
            readout=readout,
        )
        input_dim, output_dim = self.feat.shape[1], self.num_classes
        # Instantiate CF model class, load weights from original model
        self.cf_model = eval(self.cf_model_name[0].upper() + "Perturb")(
            input_dim, output_dim, self.adj, cf_model_params, beta
        )
        self.cf_model.load_state_dict(self.model.state_dict(), strict=False)

        # Freeze weights from original model in cf_model
        for name, param in self.cf_model.named_parameters():
            if name.endswith("weight") or name.endswith("bias"):
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            print("orig model requires_grad: ", name, param.requires_grad)
        for name, param in self.cf_model.named_parameters():
            print("cf model requires_grad: ", name, param.requires_grad)

    def explain_node(self, cf_optimizer, node_idx, new_idx, lr, n_momentum, num_epochs):
        self.node_idx = node_idx
        self.new_idx = new_idx

        self.x = self.feat
        self.A_x = self.adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            new_example, loss_total = self.train_node(epoch)
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1
        print("{} CF examples for node_idx = {}".format(num_cf_examples, self.node_idx))
        print(" ")
        return best_cf_example

    def explain(self, cf_optimizer, lr, n_momentum, num_epochs):

        self.x = self.feat
        self.A_x = self.adj
        self.D_x = get_degree_matrix(self.A_x)

        if cf_optimizer == "SGD" and n_momentum == 0.0:
            self.cf_optimizer = optim.SGD(self.cf_model.parameters(), lr=lr)
        elif cf_optimizer == "SGD" and n_momentum != 0.0:
            self.cf_optimizer = optim.SGD(
                self.cf_model.parameters(), lr=lr, nesterov=True, momentum=n_momentum
            )
        elif cf_optimizer == "Adadelta":
            self.cf_optimizer = optim.Adadelta(self.cf_model.parameters(), lr=lr)

        best_cf_example = []
        best_loss = np.inf
        num_cf_examples = 0
        for epoch in range(num_epochs):
            new_example, loss_total = self.train(epoch)
            if new_example != [] and loss_total < best_loss:
                best_cf_example.append(new_example)
                best_loss = loss_total
                num_cf_examples += 1
        print("{} CF examples for graph".format(num_cf_examples))
        print(" ")
        return best_cf_example

    def train_node(self, epoch):
        t = time.time()
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(**{"x": self.x, "adj": self.A_x, "edge_attr": self.edge_feat})
        output_actual, self.P = self.cf_model.forward_prediction(**{"x": self.x, "adj": self.A_x, "edge_attr": self.edge_feat})
        # Need to use new_idx from now on since adj is reindexed
        y_pred_new = torch.argmax(output[self.new_idx])
        y_pred_new_actual = torch.argmax(output_actual[self.new_idx])

        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        # (loss_total, loss_pred, loss_graph_dist, cf_adj) = self.cf_model.loss(
            # output[self.new_idx], self.y_pred_orig, y_pred_new_actual
        # )
        (loss_total, loss_pred, loss_graph_dist, cf_adj) = self.cf_model.cf_loss(
            output[self.new_idx], self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        print(
            "Node idx: {}".format(self.node_idx),
            "New idx: {}".format(self.new_idx),
            "Epoch: {:04d}".format(epoch + 1),
            "loss: {:.4f}".format(loss_total.item()),
            "pred loss: {:.4f}".format(loss_pred.item()),
            "graph loss: {:.4f}".format(loss_graph_dist.item()),
        )
        print(
            "Output: {}\n".format(output[self.new_idx].data),
            "Output nondiff: {}\n".format(output_actual[self.new_idx].data),
            "orig pred: {}, new pred: {}, new pred nondiff: {}".format(
                self.y_pred_orig, y_pred_new, y_pred_new_actual
            ),
        )
        print(" ")
        cf_stats = []
        if y_pred_new_actual != self.y_pred_orig:
            print("CF example found!")
            perturb = np.abs(cf_adj.detach().numpy() - self.adj.detach().numpy())
            perturb_edges = np.nonzero(perturb)
            if len(perturb_edges[0]) > 0:
                cf_stats = [
                    self.node_idx.item(),
                    self.new_idx.item(),
                    perturb_edges,
                    cf_adj.detach().numpy(),
                    self.adj.detach().numpy(),
                    self.y_pred_orig.item(),
                    y_pred_new.item(),
                    y_pred_new_actual.item(),
                    self.labels[self.new_idx].numpy(),
                    self.adj.shape[0],
                    loss_total.item(),
                    loss_pred.item(),
                    loss_graph_dist.item(),
                ]

        return (cf_stats, loss_total.item())


    def train(self, epoch):
        t = time.time()
        self.cf_model.train()
        self.cf_optimizer.zero_grad()

        # output uses differentiable P_hat ==> adjacency matrix not binary, but needed for training
        # output_actual uses thresholded P ==> binary adjacency matrix ==> gives actual prediction
        output = self.cf_model.forward(**{"x": self.x, "adj": self.A_x, "edge_attr": self.edge_feat})
        output_actual, self.P = self.cf_model.forward_prediction(**{"x": self.x, "adj": self.A_x, "edge_attr": self.edge_feat})
        # Need to use new_idx from now on since adj is reindexed
        y_pred_new = torch.argmax(output)
        y_pred_new_actual = torch.argmax(output_actual)
        
        # loss_pred indicator should be based on y_pred_new_actual NOT y_pred_new!
        # (loss_total, loss_pred, loss_graph_dist, cf_adj) = self.cf_model.loss(
            # output.squeeze(0), self.y_pred_orig.squeeze(0), y_pred_new_actual
        # )
        (loss_total, loss_pred, loss_graph_dist, cf_adj) = self.cf_model.cf_loss(
            output.squeeze(0), self.y_pred_orig, y_pred_new_actual
        )
        loss_total.backward()
        clip_grad_norm(self.cf_model.parameters(), 2.0)
        self.cf_optimizer.step()
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss: {:.4f}".format(loss_total.item()),
            "pred loss: {:.4f}".format(loss_pred.item()),
            "graph loss: {:.4f}".format(loss_graph_dist.item()),
        )
        print(
            "Output: {}\n".format(output.data),
            "Output nondiff: {}\n".format(output_actual.data),
            "orig pred: {}, new pred: {}, new pred nondiff: {}".format(
                self.y_pred_orig.squeeze(0), y_pred_new, y_pred_new_actual
            ),
        )
        print(" ")
        cf_stats = []
        if y_pred_new_actual != self.y_pred_orig.squeeze(0):
            print("CF example found!")
            perturb = np.abs(cf_adj.detach().numpy() - self.adj.detach().numpy())
            perturb_edges = np.nonzero(perturb)
            print("perturb_edges: {}".format(perturb_edges))
            if len(perturb_edges[0]) > 0:
                cf_stats = [
                    perturb_edges,
                    cf_adj.detach().numpy(),
                    self.adj.detach().numpy(),
                    self.y_pred_orig.squeeze(0).item(),
                    y_pred_new.item(),
                    y_pred_new_actual.item(),
                    self.labels.numpy(),
                    self.adj.shape[0],
                    loss_total.item(),
                    loss_pred.item(),
                    loss_graph_dist.item(),
                ]

        return (cf_stats, loss_total.item())

