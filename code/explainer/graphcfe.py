import random
import numpy as np
from numbers import Number
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import DenseGCNConv, DenseGraphConv
import time
import gc
from utils.gen_utils import from_edge_index_to_adj, from_adj_to_edge_index, from_adj_to_edge_index_torch, from_edge_index_to_adj_torch, get_cf_edge_mask


class GraphCFE(nn.Module):
    def __init__(self, init_params, device):
        super(GraphCFE, self).__init__()
        self.vae_type = 'graphVAE'
        self.x_dim = init_params["num_node_features"]
        self.h_dim = init_params['hidden_dim']
        self.z_dim = init_params['hidden_dim']
        self.dropout = init_params['dropout']
        self.max_num_nodes = init_params['max_num_nodes']
        self.encoder_type = 'gcn'
        self.graph_pool_type = 'mean'
        self.u_dim = 0
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)
        self.device = device

        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=self.device)
        self.prior_var = nn.Sequential(MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=self.device), nn.Sigmoid())

        # encoder
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + self.u_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU(), nn.Sigmoid())

        # decoder
        self.decoder_x = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.x_dim))
        self.decoder_a = nn.Sequential(nn.Linear(self.z_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
                                       nn.Linear(self.h_dim, self.max_num_nodes*self.max_num_nodes), nn.Sigmoid())
        self.graph_norm = nn.BatchNorm1d(self.h_dim)

    def encoder(self, features, adj, y_cf):
        # Q(Z|X,U,A,Y^CF)
        # input: x, u, A, y^cf
        # output: z
        graph_rep = self.graph_model(features, adj)  # n x num_node x h_dim
        graph_rep = self.graph_pooling(graph_rep, self.graph_pool_type)  # n x h_dim
        #graph_rep = self.graph_norm(graph_rep)
        z_mu = self.encoder_mean(torch.cat((graph_rep, y_cf), dim=1))
        z_logvar = self.encoder_var(torch.cat((graph_rep, y_cf), dim=1))
        return z_mu, z_logvar

    def get_represent(self, features, adj, y_cf):
        # encoder
        z_mu, z_logvar = self.encoder(features, adj, y_cf)

        return z_mu, z_logvar

    def decoder(self, z, y_cf):
        adj_reconst = self.decoder_a(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes,
                                                                              self.max_num_nodes)

        features_reconst = self.decoder_x(torch.cat((z, y_cf), dim=1)).view(-1, self.max_num_nodes, self.x_dim)
        return features_reconst, adj_reconst

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.sum(x, dim=1, keepdim=False)
        return out

    def prior_params(self, y_cf):  # P(Z|U)
        z_u_mu = torch.zeros((len(y_cf),self.h_dim)).to(self.device)
        z_u_logvar = torch.ones((len(y_cf),self.h_dim)).to(self.device)
        return z_u_mu, z_u_logvar


    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def score(self):
        return

    def forward(self, features, adj, y_cf):
        z_u_mu, z_u_logvar = self.prior_params(y_cf)
        # encoder
        z_mu, z_logvar = self.encoder(features, adj, y_cf)
        # reparameterize
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf)

        return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'z_u_mu': z_u_mu, 'z_u_logvar': z_u_logvar}



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


def add_list_in_dict(key, dict, elem):
    if key not in dict:
        dict[key] = [elem]
    else:
        dict[key].append(elem)
    return dict

def distance_feature(feat_1, feat_2):
    pdist = nn.PairwiseDistance(p=2)
    output = pdist(feat_1, feat_2) /4
    return output

def distance_graph_prob(adj_1, adj_2_prob):
    dist = F.binary_cross_entropy(adj_2_prob, adj_1)
    return dist

def proximity_feature(feat_1, feat_2, type='cos'):
    if type == 'cos':
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        output = cos(feat_1, feat_2)
        output = torch.mean(output)
    return output

def compute_loss(params):
    model, pred_model, batch, z_mu, z_logvar, adj_permuted, features_permuted, adj_reconst, features_reconst, \
    adj_input, features_input, y_cf, z_u_mu, z_u_logvar, z_mu_cf, z_logvar_cf = params['model'], params['pred_model'],  params['batch'], params['z_mu'], \
        params['z_logvar'], params['adj_permuted'], params['features_permuted'], params['adj_reconst'], params['features_reconst'], \
        params['adj_input'], params['features_input'], params['y_cf'], params['z_u_mu'], params['z_u_logvar'], params['z_mu_cf'], params['z_logvar_cf']

    # kl loss
    loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
    loss_kl = torch.mean(loss_kl)

    # similarity loss
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst)

    beta = 10

    loss_sim = beta * dist_x + 10 * dist_a

    # CFE loss
    y_pred = pred_model(**{'x':features_reconst.reshape(-1,features_reconst.size(-1)), 'adj':adj_reconst.reshape(-1,adj_reconst.size(-1)), 'batch':batch})  # n x num_class
    loss_cfe = F.nll_loss(y_pred, y_cf.view(-1).long())

    # rep loss
    if z_mu_cf is None:
        loss_kl_cf = 0.0
    else:
        loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
        loss_kl_cf = torch.mean(loss_kl_cf)

    loss = 1. * loss_sim + 1 * loss_kl + 1.0 * loss_cfe

    loss_results = {'loss': loss, 'loss_kl': loss_kl, 'loss_sim': loss_sim, 'loss_cfe': loss_cfe, 'loss_kl_cf':loss_kl_cf}
    return loss_results


def test(params):
    model, data_loader, pred_model, y_cf_all, dataset, metrics  = params['model'], params['data_loader'], params['pred_model'], params['y_cf'], params['dataset'], params['metrics']
    model.eval()
    pred_model.eval()
    device = model.device

    eval_results_all = {k: 0.0 for k in metrics}
    size_all = 0
    loss, loss_kl, loss_sim, loss_cfe = 0.0, 0.0, 0.0, 0.0
    batch_num = 0
    for batch_idx, data in enumerate(data_loader):
        batch_num += 1
        batch_size = len(data['y'])
        size_all += batch_size

        features = data.x_padded.float().to(device)
        adj = data.adj_padded.float().to(device)
        max_num_nodes = adj.size(1)
        features = features.reshape(-1,max_num_nodes,features.size(1))
        adj = adj.reshape(-1,max_num_nodes,max_num_nodes)
        orin_index = data.idx
        y_cf = y_cf_all[orin_index].to(device)
        y_cf = y_cf.reshape(-1,1)
        bs = len(orin_index)
        batch_padded = []
        for i in range(bs):
            batch_padded.append([i]*max_num_nodes)
        batch_padded = torch.LongTensor(batch_padded).reshape(-1,1).to(device)
        batch_padded = batch_padded.reshape(-1)
        model_return = model(features, adj, y_cf)
        adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']

        adj_reconst_binary = torch.bernoulli(adj_reconst)
        y_cf_pred = pred_model(**{'x':features_reconst.reshape(-1,features_reconst.size(-1)), 'adj':adj_reconst_binary.reshape(-1,adj_reconst_binary.size(-1)), 'batch':batch_padded})  # n x num_class
        y_pred = pred_model(**{'x':features.reshape(-1,features.size(-1)), 'adj':adj.reshape(-1,adj.size(-1)), 'batch':batch_padded})  # n x num_class

        # z_cf
        z_mu_cf, z_logvar_cf = None, None

        # compute loss
        loss_params = {'model': model, 'pred_model': pred_model, 'batch': batch_padded, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
        loss_params.update(model_return)

        loss_results = compute_loss(loss_params)
        loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch = loss_results['loss'], loss_results['loss_kl'], \
                                                                    loss_results['loss_sim'], loss_results['loss_cfe']
        loss += loss_batch
        loss_kl += loss_kl_batch
        loss_sim += loss_sim_batch
        loss_cfe += loss_cfe_batch

        # evaluate metrics
        eval_params = model_return.copy()
        eval_params.update({'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_input': adj, 'features_input': features, 'labels':data.y, 'y_pred':y_pred})

        eval_results = evaluate(eval_params)
        for k in metrics:
            eval_results_all[k] += (batch_size * eval_results[k])

    for k in metrics:
        eval_results_all[k] /= size_all

    loss, loss_kl, loss_sim, loss_cfe = loss / batch_num, loss_kl / batch_num, loss_sim / batch_num, loss_cfe / batch_num
    eval_results_all['loss'], eval_results_all['loss_kl'], eval_results_all['loss_sim'], eval_results_all['loss_cfe'] = loss, loss_kl, loss_sim, loss_cfe

    return eval_results_all


def compute_counterfactual(dataset, data, metrics, y_cf, model, pred_model, device):
    model.eval()
    pred_model.eval()

    features = data.x_padded.float().to(device)
    adj = data.adj_padded.float().to(device)
    max_num_nodes = adj.size(1)
    label = data.y.to(device)
    y_cf = y_cf.reshape(-1,1).to(device)
    features_model, adj_model = features.unsqueeze(0).to(device), adj.unsqueeze(0).to(device)
    model_return = model(features_model, adj_model, y_cf)
    adj_reconst, features_reconst = model_return['adj_reconst'], model_return['features_reconst']

    adj_reconst_binary = torch.bernoulli(adj_reconst)
    y_cf_pred = pred_model(**{'x':features_reconst.reshape(-1,features_reconst.size(-1)), 'adj':adj_reconst_binary.reshape(-1,adj_reconst_binary.size(-1))}) 
    y_pred = pred_model(**{'x':features.reshape(-1,features.size(-1)), 'adj':adj.reshape(-1,adj.size(-1))})  
    # evaluate metrics
    eval_params = {}
    eval_params.update(
        {'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_permuted': adj_model,
            'features_permuted': features_model, 'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'labels': label, 'y_pred':y_pred})

    eval_results = evaluate(eval_params)

    edge_index_cf, edge_attr_cf = from_adj_to_edge_index_torch(adj_reconst_binary[0])
    edge_mask = get_cf_edge_mask(edge_index_cf, data.edge_index)
    # The explanation is the edges that are not counterfactual edges
    return eval_results, edge_mask


def train(params):
    epochs, pred_model, model, optimizer, y_cf_all, train_loader, val_loader, test_loader, dataset, metrics = \
        params['epochs'], params['pred_model'], params['model'], params['optimizer'], params['y_cf'],\
        params['train_loader'], params['val_loader'], params['test_loader'], params['dataset'], params['metrics']
    save_model = params['save_model'] if 'save_model' in params else True
    device = model.device
    print("start training!")
    # train
    time_begin = time.time()
    best_loss = 100000

    for epoch in range(epochs + 1):
        model.train()

        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
        batch_num = 0
        for batch_idx, data in enumerate(train_loader):
            batch_num += 1
            features = data.x_padded.float().to(device)
            adj = data.adj_padded.float().to(device)
            max_num_nodes = adj.size(1)
            features = features.reshape(-1,max_num_nodes,features.size(1))
            adj = adj.reshape(-1,max_num_nodes,max_num_nodes)
            orin_index = data.idx
            y_cf = y_cf_all[orin_index]
            y_cf = y_cf.reshape(-1,1)
            bs = len(orin_index)
            batch_padded = []
            for i in range(bs):
                batch_padded.append([i]*max_num_nodes)
            batch_padded = torch.LongTensor(batch_padded).reshape(-1,1).to(device)
            batch_padded = batch_padded.reshape(-1)
            
            optimizer.zero_grad()

            # forward pass
            model_return = model(features, adj, y_cf)

            # z_cf
            z_mu_cf, z_logvar_cf = model.get_represent(model_return['features_reconst'], model_return['adj_reconst'], y_cf)

            # compute loss
            loss_params = {'model': model, 'pred_model': pred_model, 'batch': batch_padded, 'adj_input': adj, 'features_input': features, 'y_cf': y_cf, 'z_mu_cf': z_mu_cf, 'z_logvar_cf':z_logvar_cf}
            loss_params.update(model_return)

            loss_results = compute_loss(loss_params)
            loss_batch, loss_kl_batch, loss_sim_batch, loss_cfe_batch, loss_kl_batch_cf = loss_results['loss'], loss_results['loss_kl'], loss_results['loss_sim'], loss_results['loss_cfe'], loss_results['loss_kl_cf']
            loss += loss_batch
            loss_kl += loss_kl_batch
            loss_sim += loss_sim_batch
            loss_cfe += loss_cfe_batch
            loss_kl_cf += loss_kl_batch_cf

            # free up unnecessary memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()

        # backward propagation
        loss, loss_kl, loss_sim, loss_cfe, loss_kl_cf = loss / batch_num, loss_kl/batch_num, loss_sim/batch_num, loss_cfe/batch_num, loss_kl_cf/batch_num

        alpha = 5

        if epoch < 450:
            ((loss_sim + loss_kl + 0* loss_cfe)/ batch_num).backward()
        else:
            ((loss_sim + loss_kl + alpha * loss_cfe)/ batch_num).backward()
        optimizer.step()

        # free up unnecessary memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            gc.collect()

        # evaluate
        if epoch % 100 == 0:
            model.eval()
            eval_params_val = {'model': model, 'data_loader': val_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_params_tst = {'model': model, 'data_loader': test_loader, 'pred_model': pred_model, 'y_cf': y_cf_all, 'dataset': dataset, 'metrics': metrics}
            eval_results_val = test(eval_params_val)
            eval_results_tst = test(eval_params_tst)
            val_loss, val_loss_kl, val_loss_sim, val_loss_cfe = eval_results_val['loss'], eval_results_val['loss_kl'], eval_results_val['loss_sim'], eval_results_val['loss_cfe']

            metrics_results_val = ""
            metrics_results_tst = ""
            for k in metrics:
                metrics_results_val += f"{k}_val: {eval_results_val[k]:.4f} | "
                metrics_results_tst += f"{k}_tst: {eval_results_tst[k]:.4f} | "

            print(f"[Train] Epoch {epoch}: train_loss: {(loss):.4f} |" +
                  metrics_results_val + metrics_results_tst +
                  f"time: {(time.time() - time_begin):.4f} |")


def evaluate(params):
    adj_permuted, features_permuted, adj_reconst_prob, features_reconst, metrics, dataset, y_cf, y_cf_pred, labels, y_pred = \
        params['adj_permuted'], params['features_permuted'], params['adj_reconst'], \
        params['features_reconst'], params['metrics'], params['dataset'], params['y_cf'], params['y_cf_pred'], params['labels'], params['y_pred']

    adj_reconst = torch.bernoulli(adj_reconst_prob)
    eval_results = {}
    if 'causality' in metrics:
        score_causal = evaluate_causality(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst,  y_cf, labels, u)
        eval_results['causality'] = score_causal
    if 'proximity' in metrics or 'proximity_x' in metrics or 'proximity_a' in metrics:
        score_proximity, dist_x, dist_a = evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst)
        eval_results['proximity'] = score_proximity
        eval_results['proximity_x'] = dist_x
        eval_results['proximity_a'] = dist_a
    if 'validity' in metrics:
        score_valid = evaluate_validity(y_cf, y_cf_pred)
        eval_results['validity'] = score_valid
    if 'correct' in metrics:
        score_correct = evaluate_correct(dataset, adj_permuted, features_permuted, adj_reconst, features_reconst, y_cf, labels, y_cf_pred, y_pred)
        eval_results['correct'] = score_correct

    return eval_results

def evaluate_validity(y_cf, y_cf_pred):
    device = y_cf.device
    y_cf_pred_binary = F.softmax(y_cf_pred, dim=-1)
    y_cf_pred_binary = y_cf_pred_binary.argmax(dim=1).view(-1,1)
    y_eq = torch.where(y_cf == y_cf_pred_binary, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
    score_valid = torch.mean(y_eq)
    return score_valid

def evaluate_proximity(dataset, adj_permuted, features_permuted, adj_reconst_prob, adj_reconst, features_reconst):
    size = len(features_permuted)
    dist_x = torch.mean(distance_feature(features_permuted.view(size, -1), features_reconst.view(size, -1)))
    dist_a = distance_graph_prob(adj_permuted, adj_reconst_prob)
    score = dist_x + dist_a

    proximity_x = proximity_feature(features_permuted, features_reconst, 'cos')

    acc_a = (adj_permuted == adj_reconst).float().mean()
    return score, proximity_x, acc_a


def perturb_graph(adj, type='random', num_rounds=1):
    num_node = adj.shape[0]
    num_entry = num_node * num_node
    adj_cf = adj.clone()
    if type == 'random':
        # randomly add/remove edges for T rounds
        for rd in range(num_rounds):
            [row, col] = np.random.choice(num_node, size=2, replace=False)
            adj_cf[row, col] = 1 - adj[row, col]
            adj_cf[col, row] = adj_cf[row, col]

    elif type == 'IST':
        # randomly add edge
        for rd in range(num_rounds):
            idx_select = (adj_cf == 0).nonzero()  # 0
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 1
            adj_cf[col, row] = 1

    elif type == 'RM':
        # randomly remove edge
        for rd in range(num_rounds):
            idx_select = adj_cf.nonzero()  # 1
            if len(idx_select) <= 0:
                continue
            ii = np.random.choice(len(idx_select), size=1, replace=False)
            idx = idx_select[ii].view(-1)
            row, col = idx[0], idx[1]
            adj_cf[row, col] = 0
            adj_cf[col, row] = 0

    return adj_cf

def baseline_cf(dataset, data, metrics, y_cf, pred_model, device, num_rounds = 10, type='random'):
    
    features = data.x_padded.float().to(device)
    adj = data.adj_padded.float().to(device)
    max_num_nodes = adj.size(1)
    label = data.y.to(device)
    y_cf = y_cf.reshape(-1,1).to(device)
    

    adj_reconst = adj.clone()

    noise = torch.normal(mean=0.0, std=1, size=features.shape).to(device)  # add a Gaussian noise to node features
    features_reconst = features + noise

    # perturbation on A
    for t in range(num_rounds):
        adj_reconst = perturb_graph(adj_reconst, type, num_rounds=1)  # randomly perturb graph
        y_cf_pred_i = pred_model(**{'x':features_reconst.reshape(-1,features_reconst.size(-1)), 'adj':adj_reconst.reshape(-1,adj_reconst.size(-1))}).argmax(dim=1).view(-1,1)  # 1 x 1
        if y_cf_pred_i.item() == y_cf:  # Stop when f(G^CF) == Y^CF
            break

    # prediction model
    y_cf_pred = pred_model(**{'x':features_reconst.reshape(-1,features_reconst.size(-1)), 'adj':adj_reconst.reshape(-1,adj_reconst.size(-1))})
    y_pred = pred_model(**{'x':features.reshape(-1,features.size(-1)), 'adj':adj.reshape(-1,adj.size(-1))})

    # evaluate metrics
    eval_params = {}
    eval_params.update(
        {'y_cf': y_cf, 'metrics': metrics, 'y_cf_pred': y_cf_pred, 'dataset': dataset, 'adj_permuted': adj,
            'features_permuted': features, 'adj_reconst': adj_reconst, 'features_reconst': features_reconst, 'labels': label, 'y_pred':y_pred})

    eval_results = evaluate(eval_params)
    return eval_results
