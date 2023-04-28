from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_max
import os.path as osp
from torch_geometric.data import DataLoader
from explainer.explainer_utils.rcexplainer.reorganizer import *
from explainer.explainer_utils.rcexplainer.rc_train import *
import copy

class RCExplainer_Batch(torch.nn.Module):
    def __init__(self, model, device, num_labels, hidden_size, use_edge_attr=False):
        super(RCExplainer_Batch, self).__init__()
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        self.model.eval()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_edge_attr = use_edge_attr

        self.temperature = 0.1

        self.edge_action_rep_generator = Sequential(
            Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 4),
            ELU(),
            Linear(self.hidden_size * 4, self.hidden_size * 2),
            ELU(),
            Linear(self.hidden_size * 2, self.hidden_size)
        ).to(device)

        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = torch.nn.ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size * (2 + self.use_edge_attr), self.hidden_size * 2),
                ELU(),
                Linear(self.hidden_size * 2, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)
            ).to(self.device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator

    def forward(self, graph, state, train_flag=False):
        graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        if len(torch.where(state==True)[0]) == 0:
            subgraph_rep = torch.zeros(graph_rep.size()).to(self.device)
        else:
            subgraph = relabel_graph(graph, state)
            subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        ava_edge_index = graph.edge_index.T[~state].T
        ava_edge_attr = graph.edge_attr[~state]
        ava_node_reps = self.model.get_emb(graph.x, ava_edge_index, ava_edge_attr, graph.batch)

        if self.use_edge_attr:
            ava_edge_reps = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                         ava_node_reps[ava_edge_index[1]],
                                         ava_edge_reps], dim=1).to(self.device)
        else:

            ava_action_reps = torch.cat([ava_node_reps[ava_edge_index[0]],
                                         ava_node_reps[ava_edge_index[1]]], dim=1).to(self.device)
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps)

        ava_action_batch = graph.batch[ava_edge_index[0]]
        ava_y_batch = graph.y[ava_action_batch]

        unique_batch, ava_action_batch = torch.unique(ava_action_batch, return_inverse=True)

        ava_action_probs = self.predict_star(graph_rep, subgraph_rep, ava_action_reps, ava_y_batch, ava_action_batch)

        # assert len(ava_action_probs) == sum(~state)

        added_action_probs, added_actions = scatter_max(ava_action_probs, ava_action_batch)

        if train_flag:
            rand_action_probs = torch.rand(ava_action_probs.size()).to(self.device)
            _, rand_actions = scatter_max(rand_action_probs, ava_action_batch)

            return ava_action_probs, ava_action_probs[rand_actions], rand_actions, unique_batch

        return ava_action_probs, added_action_probs, added_actions, unique_batch

    def predict_star(self, graph_rep, subgraph_rep, ava_action_reps, target_y, ava_action_batch):
        action_graph_reps = graph_rep - subgraph_rep
        action_graph_reps = action_graph_reps[ava_action_batch]
        action_graph_reps = torch.cat([ava_action_reps, action_graph_reps], dim=1)

        action_probs = []
        for i_explainer in self.edge_action_prob_generator:
            i_action_probs = i_explainer(action_graph_reps)
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=1)

        action_probs = action_probs.gather(1, target_y.view(-1,1))
        action_probs = action_probs.reshape(-1)

        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5, scope='all'):
        if scope in ['all']:
            params = self.parameters()
        else:
            params = list(self.edge_action_rep_generator.parameters()) + \
                     list(self.edge_action_prob_generator.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer




def train_rcexplainer(rc_explainer, train_dataset, test_dataset, loader, batch_size, lr, weight_decay, topk_ratio):
    train_loader = loader['train']
    test_loader = loader['test']
    model = rc_explainer.model
    device = rc_explainer.device
    train_dataset, train_loader = filter_correct_data_batch(model, train_dataset, train_loader, 'training',
                                                                batch_size=batch_size)
    test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader, 'testing',
                                                              batch_size=1)
    optimizer = rc_explainer.get_optimizer(lr=lr, weight_decay=weight_decay)
    rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec = train_policy(rc_explainer, model, device,
                                                                                  train_loader, test_loader, optimizer, batch_size=batch_size, topK_ratio=topk_ratio)
    return rc_explainer

