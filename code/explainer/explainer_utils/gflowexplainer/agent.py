'''Adapted from https://github.com/bengioe/gflownet'''
import copy
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch

M = 1e3

class GraphAgent(nn.Module):
    
    def __init__(self, n_conv, n_hidden, n_out_stem, n_out_graph, n_input):
        super().__init__()
        # self.linear_layers = nn.ModuleList([
        #     gnn.Linear(-1, n_hidden), gnn.Linear(-1, n_hidden), gnn.Linear(-1, n_hidden)])

        self.conv = nn.ModuleList()
        self.conv.append(gnn.GCNConv(in_channels=n_input, out_channels=n_hidden))
        for i in range(n_conv-2):
            self.conv.append(gnn.GCNConv(in_channels=n_hidden, out_channels=n_hidden))
        self.conv.append(gnn.GCNConv(in_channels=n_hidden, out_channels=n_hidden))
        self.edge_emb = gnn.Linear(-1, n_hidden)

        self.edge_action_rep = nn.Sequential(gnn.Linear(-1, n_hidden),
                                             nn.ReLU()
                                             )
        self.edge_action_prob = nn.Sequential(gnn.Linear(-1, n_hidden),
                                              nn.ReLU(),
                                              nn.Linear(n_hidden, 1),
                                              nn.Sigmoid()
                                              )
        
        self.escort_p = 6
        self.training_steps = 0
        self.n_conv = n_conv
        self.n_hidden = n_hidden
        self.categorical_style = 'softmax'

    def forward(self, graph, state, full_graph, gnn_model, actions=None):
        E = full_graph.num_edges
        self.device = graph.x.device
        full_edge_attr = self.edge_emb(full_graph.edge_attr) #(n,n_hidden)
        out = graph.x
        for i in range(self.n_conv):
            out = F.relu(self.conv[i](out, graph.edge_index.long())) #(mbsize *n_node,n_hidden)

        graph_reps = gnn_model.get_graph_rep(graph).detach() #(mbsize, n_hidden)
        full_graph_rep = gnn_model.get_graph_rep(full_graph).detach() #(1, n_hidden)

        N = int(out.size(0) / full_graph.num_nodes)
        edge_reps = self.edge_action_rep(
            torch.cat([
                out[full_graph.edge_index[0].repeat(1, N).squeeze(0)].to(self.device),
                out[full_graph.edge_index[1].repeat(1, N).squeeze(0)].to(self.device),
                full_edge_attr.to(self.device).repeat(N, 1)], dim=1))  #(mbsize * num_node, n_hidden)
        edge_reps_list = [edge_reps[E * i : E * (i + 1)] for i in range(N)]

        if actions is None:
            if isinstance(state, (list, tuple)):
                selections = state
            else:
                selections = [state]
        else:
            selections = []
            for action in actions:
                selection = torch.zeros(E).bool()
                selection[action] = True
                selections.append(selection)
        edge_preds = self.estimate_edge_selection_prob(full_graph_rep, graph_reps, edge_reps_list, selections)
        
        # free up unnecessary memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()
        return edge_preds, None

    def estimate_edge_selection_prob(self, graph_rep, subgraph_reps, edge_reps_list, selections):
        '''
        selections: list of selection (zero vector with only one True)
        Return: list of [-M,-M, ..., p, -M, -M]
        list_len = len(selections). p is the probability of the selected edge action
        (all current edges are candidate action for no action input)

        '''
        action_probs = []
        for (subgraph_rep, edge_reps, selection) in zip(subgraph_reps, edge_reps_list, selections):
            # _action_prob = self.edge_action_prob(edge_reps)
            _action_prob = self.edge_action_prob(
                torch.cat([
                    graph_rep.repeat(edge_reps.size(0), 1).to(self.device), 
                    subgraph_rep.repeat(edge_reps.size(0), 1).to(self.device),
                    edge_reps.to(self.device)
                ], dim=-1).to(self.device))
                
            _action_prob = _action_prob.view(-1)
            action_prob = -M * torch.ones_like(_action_prob)
            action_prob[selection] = _action_prob[selection]
            action_probs.append(action_prob)

        return action_probs

    def foward_multisteps(self, graph, gnn_model, remove_ratio=1):
        exp_graph = copy.deepcopy(graph)
        exp_graph = exp_graph.to(self.device)
        graph = graph.to(self.device)
        state = torch.ones(graph.num_edges).bool()
        edge_imp = graph.num_edges * torch.ones(graph.num_edges)

        n_edge_remove = int(remove_ratio * graph.num_edges - 1) # int(remove_ratio * graph.num_edges)
        edge_index = copy.copy(exp_graph.edge_index)
        edge_attr = copy.copy(exp_graph.edge_attr)
        for i in range(n_edge_remove):
            edge_preds, _ = self.forward(exp_graph, state, graph, gnn_model)
            edge_id = torch.argmax(edge_preds[0], dim=-1)
            exp_graph.edge_index = edge_index[:, state]
            exp_graph.edge_attr = edge_attr[state]
            state[edge_id] = False
            edge_imp[edge_id] = i
        return exp_graph, edge_imp   #return length: removed num; edge_imp: indicate the i-th removal

    def index_output_by_action(self, graph, edge_out, graph_out, action):
        return torch.cat([out[a] for (out, a) in zip(edge_out, action)])

    def sum_output(self, s, edge_out, graph_out):
        return edge_out.sum(dim=1)# + graph_out

    def out_to_policy(self, s, stem_o, graph_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o)
            graph_e = torch.exp(graph_o[:, 0])
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            graph_e = abs(graph_o[:, 0])**self.escort_p
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + graph_e + 1e-8
        return graph_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, edge_out, graph_out):
        graph_p, stem_p = self.out_to_policy(s, edge_out, graph_out)
        graph_lsm = torch.log(graph_p + 1e-20)
        stem_lsm = torch.log(stem_p + 1e-20)
        return -self.index_output_by_action(s, stem_lsm, graph_lsm, a)


def graph2data(graph, mdp, bonds=False, nblocks=False):
    f = lambda x: torch.tensor(x, dtype=torch.long, device=mdp.device)
    if len(graph.blockidxs) == 0:
        # There's an extra block embedding for the empty graphecule
        data = Data(
            x=f([mdp.num_true_blocks]),
            edge_index=f([[],[]]),
            edge_attr=f([]).reshape((0,2)),
            stems=f([(0,0)]),
            stemtypes=f([mdp.num_stem_types])) # also extra stem type embedding
        return data
    edges = [(i[0], i[1]) for i in graph.jbonds]
    t = mdp.true_blockidx
    edge_attrs = [(mdp.stem_type_offset[t[graph.blockidxs[i[0]]]] + i[2],
                    mdp.stem_type_offset[t[graph.blockidxs[i[1]]]] + i[3])
                    for i in graph.jbonds]


    stemtypes = [mdp.stem_type_offset[t[graph.blockidxs[i[0]]]] + i[1] for i in graph.stems]
    data = Data(x=f([t[i] for i in graph.blockidxs]),
                edge_index=f(edges).T if len(edges) else f([[],[]]),
                edge_attr=f(edge_attrs) if len(edges) else f([]).reshape((0,2)),
                stems=f(graph.stems) if len(graph.stems) else f([(0,0)]),
                stemtypes=f(stemtypes) if len(graph.stems) else f([mdp.num_stem_types]))
    data.to(mdp.device)
    assert not bonds and not nblocks
    return data


def graphs2batch(graphs, mdp):
    batch = Batch.from_data_list(graphs, 
                                 follow_batch=['stems'])
    batch.to(mdp.device)
    return batch


def create_agent(args, device):
    n_input = args.n_input
    return GraphAgent(args.n_conv, args.n_hidden,
                      args.n_out_stem, args.n_out_graph, n_input).to(device)