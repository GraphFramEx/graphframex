import torch
import torch.nn.functional as F

import copy
import numpy as np
import networkx as nx
from tqdm import tqdm

import gym
from gym import spaces
from gym.utils import seeding

from torch_geometric.data import Data, DataLoader, Batch

EPS = 1e-15

def bool2str(s): 
    s = s.detach().cpu().numpy()
    s = ''.join([str(int(i)) for i in s])
    return s

class GraphMDP(gym.Env):
    '''Markov decision process for graph data'''
    def __init__(self, graph, target_model, device,
                 cf_flag=False, seed=42):
        '''
        params:
            graph (pyg format):  graph data
        '''
        super(GraphMDP, self).__init__()
        self.graph = graph
        self.device = device
        self.target_model = target_model
        self.n_max_actions = graph.num_edges

        self.connectivity = {}
        self.state = torch.ones(graph.num_edges).long()
        self.action_space = spaces.Discrete(self.n_max_actions)
        self.seed(seed)
        
        self.cf_flag = cf_flag
        self.target_pred = F.softmax(target_model(graph)).detach()
        self.target_y = self.target_pred.argmax(dim=-1)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def __mask__(self):
        mask = self.state.detach().cpu().numpy()
        return mask.astype('int8')
    
    def _state2graph(self, state=None):
        if state is None:
            state = self.state
        if state.float().sum() == 0:
            raise ValueError
        state = state.bool()
        edge_index = self.graph.edge_index[:, state].long()
        edge_attr = self.graph.edge_attr[state]
        x = self.graph.x
        # x, edge_index, batch, _ = self._relabel(self.graph, edge_index)
        return Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=torch.zeros(x.size(0), dtype=torch.int64))
    
    def states2data_list(self, states):
        if isinstance(states, torch.Tensor):
            states = [states]
        data_list = []
        for state in states:
            graph = self._state2graph(state)
            data_list.append(graph.to(self.device))
        return data_list

    def states2batch(self, states):
        data_list = self.states2data_list(states)
        return Batch.from_data_list(data_list)

    def _relabel(self, graph, edge_index):
        '''Remove nodes with zero degree and re-organize the node id'''
        sub_nodes = torch.unique(edge_index)
        x = graph.x[sub_nodes]
        batch = graph.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = graph.pos[sub_nodes]
        except:
            pass
        # remapping the nodes in the subgraph to new ids.
        node_idx = row.new_full((graph.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos
    
    def _get_info(self, pred):
        return {'distance': 1 - pred[self.target_y], 'last_state': self._last_state}

    def _get_reward(self, state, pre_reward=0, mode='mutual_info'):
        '''Compute the rewards for the generated subgraphs'''
        graph = self._state2graph(state).to(self.device)
        out = self.target_model(graph)
        pred = F.softmax(out, dim=-1).detach()
        if self.cf_flag:
            # for counterfactual explanations, we use the negative logits at the target class as reward
            reward = torch.log(2 - pred)[:, self.target_y]
        else:
            # following RC-Explainer, we use three modes of rewards
            if mode == 'mutual_info':
                reward = torch.sum(self.target_pred * torch.log(pred + 1), dim=1)
                reward += 2 * (self.target_y == pred.argmax(dim=1)).float()
            elif mode == 'binary':
                reward = 2 * (self.target_y == pred.argmax(dim=1)).float()
            elif mode == 'cross_entropy':
                reward = torch.log(pred + 1)[:, self.target_y]
        return reward

    def is_connected(self, state):
        _key = bool2str(state)
        if not (_key in self.connectivity.keys()):
            if state.float().sum():
                G = nx.Graph()
                _edges = self.graph.edge_index[:, state.bool()]
                _, _edges, _, _ = self._relabel(self.graph, _edges)
                G.add_edges_from(_edges.cpu().numpy().T)
                self.connectivity[_key] = nx.is_connected(G)
            else:
                self.connectivity[_key] = True
        if self.connectivity[_key]:
            return True
        else:
            return False

    def reset(self):
        self.state = torch.zeros(self.graph.num_edges).long()
        observed = np.zeros(self.num_states, dtype=np.float32)
        observed[self.state] = 1.0

        return observed

    def sample_actions(self, connected=False):
        while True:
            action = self.action_space.sample(self.__mask__())
            tmp_state = copy.deepcopy(self.state)
            tmp_state[action] = 0
            if (not connected) or \
                (connected and self.is_connected(tmp_state)):
                break
        return action

    def action2candidates(self, actions):
        results = []
        for action in actions:
            state = copy.deepcopy(self.state)
            state[action] = 0
            reward = self._get_reward(state)
            done = 1 if state.sum()==len(state) else 0
            results.append((self.state, reward, action, state.long(), done))
        return results

    def step(self, action):
        '''
        params:
            action (int):  id of the chosen edge
        '''
        state_next = copy.deepcopy(self.state)
        state_next[action] = 0
        reward = self._get_reward(state_next)
        
        self._last_state = self.state
        self.state = state_next.long()
        return self.state, action, reward

    def parents(self, state=None, connected=True):
        '''Return the parent state(s)'''
        if connected: 
            assert self.is_connected(state) == True
        _parents, _actions = [], []
        is_final = state.float().sum() == 0
        state = self.state if state is None else state
        reward = self._get_reward(state)
        assert state.float().sum() > 0
        for idx in np.arange(len(state))[(~state).bool()]:
            tmp_state = copy.deepcopy(state)
            tmp_state[idx] = 1
            tmp_state = tmp_state.bool()
            if (not connected) or \
                (connected and self.is_connected(tmp_state)):
                _parents.append(tmp_state)
                _actions.append(idx)
        return _parents, _actions, reward, state, is_final


def create_mdps(dataset, target_model, device):
    mdps = []
    data_loader = DataLoader(dataset, batch_size=1)
    for graph in tqdm(data_loader):
        graph.to(device)
        mdps.append(GraphMDP(graph, target_model, device))
    return mdps