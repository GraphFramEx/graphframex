import random
from numbers import Number
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch_geometric.nn import DenseGCNConv, DenseGraphConv


class GraphCFE(nn.Module):
    def __init__(self, params):
        super(GraphCFE, self).__init__()
        self.vae_type = 'graphVAE'
        self.x_dim = params["num_node_features"]
        self.h_dim = params['hidden_dim']
        self.z_dim = 16
        self.dropout = params['dropout']
        self.max_num_nodes = params['max_num_nodes']
        self.encoder_type = 'gcn'
        self.graph_pool_type = 'mean'
        self.u_dim = 0
        if self.encoder_type == 'gcn':
            self.graph_model = DenseGCNConv(self.x_dim, self.h_dim)
        elif self.encoder_type == 'graphConv':
            self.graph_model = DenseGraphConv(self.x_dim, self.h_dim)

        # prior
        self.prior_mean = MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device)
        self.prior_var = nn.Sequential(MLP(self.u_dim, self.z_dim, self.h_dim, n_layers=1, activation='none', slope=.1, device=device), nn.Sigmoid())

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

    def decoder(self, z, y_cf, u):
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
        # encoder
        z_mu, z_logvar = self.encoder(features, adj, y_cf)
        # reparameterize
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst = self.decoder(z_sample, y_cf, u_onehot)

        return {'z_mu': z_mu, 'z_logvar': z_logvar, 'adj_permuted': adj, 'features_permuted': features,
                'adj_reconst': adj_reconst, 'features_reconst': features_reconst}


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
                    logits = self.model(data=data)
                    emb = self.model.get_emb(data=data)
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
                        data.edge_attr,
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
                logits = self.model(data=data)
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
                        x, edge_index, y, subset, mask, kwargs = self.get_subgraph(
                            node_idx=node_idx,
                            x=data.x,
                            edge_index=data.edge_index,
                            y=data.y,
                        )
                        edge_attr = data.edge_attr[mask]
                        emb = self.model.get_emb(x, edge_index, edge_attr)
                        new_node_index = int(torch.where(subset == node_idx)[0])
                    pred, edge_mask = self.explain(
                        x,
                        edge_index,
                        edge_attr,
                        emb,
                        tmp,
                        training=True,
                        node_idx=new_node_index,
                    )
                    loss_tmp = self.__loss__(pred[new_node_index], pred_dict[node_idx])
                    loss_tmp.backward()
                    loss += loss_tmp.item()

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f"Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}")
            print(f"training time is {duration:.5}s")


