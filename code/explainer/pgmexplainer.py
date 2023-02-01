import numpy as np
import pandas as pd
import torch
from pgmpy.estimators.CITests import chi_square
from scipy.special import softmax
from torch_geometric.utils import k_hop_subgraph

###### Node Classification ######


class Node_Explainer:
    def __init__(
        self,
        model,
        edge_index,
        edge_attr,
        X,
        num_layers,
        device=None,
        mode=0,
        print_result=1,
    ):
        self.model = model
        self.model.eval()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.X = X
        self.num_layers = num_layers
        self.device = device
        self.mode = mode
        self.print_result = print_result

    def perturb_features_on_node(self, feature_matrix, node_idx, random=0, mode=0):
        # return a random perturbed feature matrix
        # random = 0 for nothing, 1 for random.
        # mode = 0 for random 0-1, 1 for scaling with original feature

        X_perturb = feature_matrix
        if mode == 0:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.random.randint(2, size=X_perturb[node_idx].shape[0])
            X_perturb[node_idx] = perturb_array
        elif mode == 1:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.multiply(
                    X_perturb[node_idx],
                    np.random.uniform(
                        low=0.0, high=2.0, size=X_perturb[node_idx].shape[0]
                    ),
                )
            X_perturb[node_idx] = perturb_array
        return X_perturb

    def explain(
        self,
        node_idx,
        target,
        num_samples=100,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1,
    ):
        neighbors, _, _, _ = k_hop_subgraph(node_idx, self.num_layers, self.edge_index)
        neighbors = neighbors.cpu().detach().numpy()

        if node_idx not in neighbors:
            neighbors = np.append(neighbors, node_idx)

        pred_torch = self.model(self.X, self.edge_index, self.edge_attr).cpu()
        soft_pred = np.asarray(
            [
                softmax(np.asarray(pred_torch[node_].data))
                for node_ in range(self.X.shape[0])
            ]
        )

        pred_node = np.asarray(pred_torch[node_idx].data)
        label_node = np.argmax(pred_node)
        soft_pred_node = softmax(pred_node)

        Samples = []
        Pred_Samples = []

        for iteration in range(num_samples):

            X_perturb = self.X.cpu().detach().numpy()
            sample = []
            for node in neighbors:
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(
                        X_perturb, node, random=seed
                    )
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float).to(self.device)
            pred_perturb_torch = self.model(
                X_perturb_torch, self.edge_index, self.edge_attr
            ).cpu()
            soft_pred_perturb = np.asarray(
                [
                    softmax(np.asarray(pred_perturb_torch[node_].data))
                    for node_ in range(self.X.shape[0])
                ]
            )

            sample_bool = []
            for node in neighbors:
                if (soft_pred_perturb[node, target] + pred_threshold) < soft_pred[
                    node, target
                ]:
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples - Samples
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [
                    Samples[s, i] * 10 + Pred_Samples[s, i] + 1
                    for i in range(Samples.shape[1])
                ]
            )

        data_pgm = pd.DataFrame(Combine_Samples)
        data_pgm = data_pgm.rename(
            columns={0: "A", 1: "B"}
        )  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data_pgm.columns)))

        p_values = []
        for node in neighbors:
            if node == node_idx:
                p = 0  # p<0.05 => we are confident that we can reject the null hypothesis (i.e. the prediction is the same after perturbing the neighbouring node
                # => this neighbour has no influence on the prediction - should not be in the explanation)
            else:
                chi2, p, _ = chi_square(
                    ind_ori_to_sub[node],
                    ind_ori_to_sub[node_idx],
                    [],
                    data_pgm,
                    boolean=False,
                    significance_level=0.05,
                )
            p_values.append(p)

        pgm_stats = dict(zip(neighbors, p_values))
        return pgm_stats


####### Graph Classification #######


def n_hops_A(A, n_hops):
    # Compute the n-hops adjacency matrix
    adj = torch.tensor(A, dtype=torch.float)
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj
        prev_hop_adj = hop_adj
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.numpy().astype(int)


class Graph_Explainer:
    def __init__(
        self,
        model,
        edge_index,
        edge_attr,
        X,
        num_layers=None,
        device=None,
        perturb_feature_list=None,
        perturb_mode="mean",  # mean, zero, max or uniform
        perturb_indicator="diff",  # diff or abs
        print_result=1,
        snorm_n=None,
        snorm_e=None,
    ):
        self.model = model
        self.model.eval()
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.X_feat = X.cpu().numpy()
        self.device = device
        self.snorm_n = snorm_n
        self.snorm_e = snorm_e
        self.num_layers = num_layers
        self.perturb_mode = perturb_mode
        self.perturb_indicator = perturb_indicator
        self.print_result = print_result

    def perturb_features_on_node(self, feature_matrix, node_idx, random=0):

        X_perturb = feature_matrix.copy()
        perturb_array = X_perturb[node_idx].copy()
        epsilon = 0.05 * np.max(self.X_feat, axis=0)
        seed = np.random.randint(2)

        if random == 1:
            if seed == 1:
                for i in range(perturb_array.shape[0]):
                    if self.perturb_mode == "mean":
                        perturb_array[i] = np.mean(feature_matrix[:, i])
                    elif self.perturb_mode == "zero":
                        perturb_array[i] = 0
                    elif self.perturb_mode == "max":
                        perturb_array[i] = np.max(feature_matrix[:, i])
                    elif self.perturb_mode == "uniform":
                        perturb_array[i] = perturb_array[i] + np.random.uniform(
                            low=-epsilon[i], high=epsilon[i]
                        )
                        if perturb_array[i] < 0:
                            perturb_array[i] = 0
                        elif perturb_array[i] > np.max(self.X_feat, axis=0)[i]:
                            perturb_array[i] = np.max(self.X_feat, axis=0)[i]

        X_perturb[node_idx] = perturb_array

        return X_perturb

    def batch_perturb_features_on_node(
        self, num_samples, index_to_perturb, percentage, p_threshold, pred_threshold
    ):
        X_torch = torch.tensor(self.X_feat, dtype=torch.float).to(self.device)
        pred_torch = self.model(X_torch, self.edge_index, self.edge_attr).cpu()
        soft_pred = np.asarray(softmax(np.asarray(pred_torch[0].data)))
        pred_label = np.argmax(soft_pred)
        num_nodes = self.X_feat.shape[0]
        Samples = []
        for iteration in range(num_samples):
            X_perturb = self.X_feat.copy()
            sample = []
            for node in range(num_nodes):
                if node in index_to_perturb:
                    seed = np.random.randint(100)
                    if seed < percentage:
                        latent = 1
                        X_perturb = self.perturb_features_on_node(
                            X_perturb, node, random=latent
                        )
                    else:
                        latent = 0
                else:
                    latent = 0
                sample.append(latent)

            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float)
            pred_perturb_torch = self.model(
                X_perturb_torch, self.edge_index, self.edge_attr
            ).cpu()
            soft_pred_perturb = np.asarray(
                softmax(np.asarray(pred_perturb_torch[0].data))
            )
            pred_change = np.max(soft_pred) - soft_pred_perturb[pred_label]

            sample.append(pred_change)
            Samples.append(sample)

        Samples = np.asarray(Samples)
        if self.perturb_indicator == "abs":
            Samples = np.abs(Samples)

        top = int(num_samples / 8)
        top_idx = np.argsort(Samples[:, num_nodes])[-top:]
        for i in range(num_samples):
            if i in top_idx:
                Samples[i, num_nodes] = 1
            else:
                Samples[i, num_nodes] = 0

        return Samples

    def explain(
        self,
        num_samples=1000,
        percentage=10,
        top_node=None,
        p_threshold=0.05,
        pred_threshold=0.1,
    ):

        num_nodes = self.X_feat.shape[0]
        if top_node == None:
            top_node = int(num_nodes * 0.3)

        #         Round 1
        Samples = self.batch_perturb_features_on_node(
            int(num_samples / 2),
            range(num_nodes),
            percentage,
            p_threshold,
            pred_threshold,
        )

        data = pd.DataFrame(Samples)

        p_values = []
        candidate_nodes = []

        target = (
            num_nodes  # The entry for the graph classification data is at "num_nodes"
        )
        for node in range(num_nodes):
            chi2, p, _ = chi_square(
                node, target, [], data, boolean=False, significance_level=0.05
            )
            p_values.append(p)

        number_candidates = top_node
        candidate_nodes = np.argpartition(p_values, number_candidates)[
            0:number_candidates
        ]

        #         Round 2
        Samples = self.batch_perturb_features_on_node(
            num_samples, candidate_nodes, percentage, p_threshold, pred_threshold
        )
        data = pd.DataFrame(Samples)

        p_values = []
        dependent_nodes = []

        target = num_nodes
        for node in range(num_nodes):
            chi2, p, _ = chi_square(
                node, target, [], data, boolean=False, significance_level=0.05
            )
            p_values.append(p)
            if p < p_threshold:
                dependent_nodes.append(node)

        top_p = np.min((top_node, num_nodes - 1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = list(ind_top_p)

        pgm_stats = dict(zip(pgm_nodes, p_values))
        return pgm_stats
