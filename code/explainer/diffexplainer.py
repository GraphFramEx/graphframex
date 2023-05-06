import logging
import time
import os
import numpy as np
import torch
import math
# from .visual import *
import gc

EPS = 1e-6
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from explainer.explainer_utils.diffexplainer.graph_utils import graph2tensor, tensor2graph, gen_list_of_data_single, generate_mask, gen_full
from explainer.explainer_utils.diffexplainer.pgnn import Powerful

import wandb

def diff_parse_args(parser):
    parser.add_argument('--root', type=str, default="results/distribution/",
                        help='Result directory.')
    
    parser.add_argument('--task', type=str, default="gc")
    parser.add_argument('--normalization', type=str, default="instance")
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--num_layers_diff', type=int, default=6)
    parser.add_argument('--layers_per_conv', type=int, default=1)
    parser.add_argument('--train_batchsize', type=int, default=16)
    parser.add_argument('--test_batchsize', type=int, default=16)
    parser.add_argument('--sigma_length', type=int, default=10)
    parser.add_argument('--epoch_diff', type=int, default=100)
    parser.add_argument('--feature_in', type=int)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--data_size', type=int, default=-1)

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--alpha_cf', type=float, default=0.5)
    parser.add_argument('--dropout_diff', type=float, default=0.001)
    parser.add_argument('--learning_rate_diff', type=float, default=1e-3)
    parser.add_argument('--lr_decay_diff', type=float, default=0.999)
    parser.add_argument('--weight_decay_diff', type=float, default=0)
    parser.add_argument('--prob_low', type=float, default=0.0)
    parser.add_argument('--prob_high', type=float, default=0.4)
    parser.add_argument('--sparsity_level', type=float, default=2.5)

    parser.add_argument('--cat_output', type=bool, default=True)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--noise_mlp', type=bool, default=True)
    parser.add_argument('--simplified', type=bool, default=False)
    return parser.parse_args()


def loss_func_bce(score_list, groundtruth, sigma_list, mask, device, sparsity_level):
    '''
    params:
        score_list: [len(sigma_list)*bsz, N, N]
        groundtruth: [len(sigma_list)*bsz, N, N]
        mask:[len(sigma_list)*bsz, N, N]
    '''
    bsz = int(score_list.size(0) / len(sigma_list))
    num_node = score_list.size(-1)
    score_list = score_list * mask
    groundtruth = groundtruth * mask
    pos_weight = torch.full([num_node * num_node], sparsity_level).to(device)
    BCE = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    score_list_ = torch.flatten(score_list, start_dim=1, end_dim=-1)
    groundtruth_ = torch.flatten(groundtruth, start_dim=1, end_dim=-1)
    loss_matrix = BCE(score_list_, groundtruth_)
    loss_matrix = loss_matrix.view(groundtruth.size(0), num_node, num_node)
    loss_matrix = loss_matrix * (
                1 - 2 * torch.tensor(sigma_list).repeat(bsz).unsqueeze(-1).unsqueeze(-1).expand(groundtruth.size(0), num_node, num_node).to(device)
                + 1.0 / len(sigma_list))
    loss_matrix = loss_matrix * mask
    loss_matrix = (loss_matrix + torch.transpose(loss_matrix, -2, -1)) / 2
    loss = torch.mean(loss_matrix)
    return loss

def loss_cf_exp(gnn_model, graph_batch, score, y_pred, y_exp, full_edge, mask, ds, task="nc"):
    score_tensor = torch.stack(score, dim=0).squeeze(-1)
    score_tensor = torch.mean(score_tensor, dim=0).view(-1,1)
    mask_bool = mask.bool().view(-1,1)
    edge_mask_full = score_tensor[mask_bool]
    assert edge_mask_full.size(0) == full_edge.size(1)
    criterion = torch.nn.NLLLoss()
    if task == "nc":
        output_prob_cont, output_repr_cont = gnn_model(x=graph_batch.x, edge_index=full_edge, edge_mask=edge_mask_full) #mapping=graph_batch.mapping)
    else:
        output_prob_cont = gnn_model(x=graph_batch.x, edge_index=full_edge,
                                                                        edge_weight=edge_mask_full,
                                                                        batch=graph_batch.batch)
        output_repr_cont = gnn_model.get_graph_rep(x=graph_batch.x, edge_index=full_edge, edge_weight=edge_mask_full, batch=graph_batch.batch)
        
    n = output_repr_cont.size(-1)
    bsz = output_repr_cont.size(0)
    y_exp = output_prob_cont.argmax(dim=-1)
    inf_diag = torch.diag(-torch.ones((n)) / 0).unsqueeze(0).repeat(bsz, 1, 1).to(y_pred.device)
    neg_prop = (output_repr_cont.unsqueeze(1).expand(bsz, n, n) + inf_diag).logsumexp(-1) - output_repr_cont.logsumexp(-1).unsqueeze(1).repeat(1, n)
    loss_cf = criterion(neg_prop, y_pred)
    labels = torch.LongTensor([[i] for i in y_pred]).to(y_pred.device)
    fid_drop = (1- output_prob_cont.gather(1, labels).view(-1)).detach().cpu().numpy()
    fid_drop = np.mean(fid_drop)
    acc_cf = float(y_exp.eq(y_pred).sum().item() / y_pred.size(0)) #less, better
    return loss_cf, fid_drop, acc_cf

def model_save(args, model):
    exp_dir = f'{args.root}/{args.dataset_name}/'
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(model, os.path.join(exp_dir, "best_model.pth"))
    print(f"save model to {exp_dir}/best_model.pth")











class Explainer(object):

    def __init__(self, model, device):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__
        self.last_result = None
        self.vis_dict = None

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def __relabel__(self, g, edge_index):

        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos

    def __reparameterize__(self, log_alpha, beta=0.1, training=True):

        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def pack_explanatory_subgraph(self, top_ratio=0.2,
                                  graph=None, imp=None, relabel=True):

        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, 'length mismatch'

        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        if relabel:
            exp_subgraph.x, exp_subgraph.edge_index, exp_subgraph.batch, exp_subgraph.pos = self.__relabel__(
                exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph

    def evaluate_recall(self, topk=10):

        graph, imp = self.last_result
        E = graph.num_edges
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        return float(values.sum()) / float(graph.ground_truth_mask.sum())

    def evaluate_acc(self, top_ratio_list, graph=None, imp=None):
        if graph is None:
            assert self.last_result is not None
            graph, imp = self.last_result
        acc = np.array([[]])
        prob = np.array([[]])
        y = graph.y
        for idx, top_ratio in enumerate(top_ratio_list):
            if top_ratio == 1.0:
                self.model(graph)
            else:
                exp_subgraph = self.pack_explanatory_subgraph(top_ratio,
                                                              graph=graph, imp=imp)
                self.model(exp_subgraph)
            res_acc = (y == self.model.readout.argmax(dim=1)).detach().cpu().float().view(-1, 1).numpy()
            res_prob = self.model.readout[0, y].detach().cpu().float().view(-1, 1).numpy()
            acc = np.concatenate([acc, res_acc], axis=1)
            prob = np.concatenate([prob, res_prob], axis=1)
        return acc, prob




class DiffExplainer(Explainer):
    def __init__(self, model, device):
        super(DiffExplainer, self).__init__(model, device)
    def explain_graph_task(self, args, train_dataset, test_dataset):
        gnn_model = self.model.to(self.device)
        model = Powerful(args, self.device).to(self.device)
        self.train(args, model, gnn_model, train_dataset, test_dataset)

    def train(self, args, model, gnn_model, train_dataset, test_dataset):
        best_sparsity = np.inf
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate_diff,
                                     betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=args.weight_decay_diff)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_diff)
        noise_list = args.noise_list
        for epoch in range(args.epoch_diff):
            print(f"start epoch {epoch}")
            train_losses = []
            train_loss_dist = []
            train_loss_cf = []
            train_acc = []
            train_fid = []
            train_sparsity = []
            train_remain = []
            model.train()
            train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
            for i, graph in enumerate(train_loader):
                if graph.is_directed() == True:
                    edge_index_temp = graph.edge_index
                    graph.edge_index = to_undirected(edge_index=edge_index_temp)
                graph.to(self.device)
                train_adj_b, train_x_b = graph2tensor(graph, device=self.device)
                # train_adj_b: [bsz, N, N]; train_x_b: [bsz, N, C]
                sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length)) \
                    if noise_list is None else noise_list
                train_node_flag_b = train_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32) #[bsz, N]
                # all nodes that are not connected with others
                if isinstance(sigma_list, float):
                    sigma_list = [sigma_list]
                train_x_b, train_ori_adj_b, train_node_flag_sigma, train_noise_adj_b, noise_diff =  \
                    gen_list_of_data_single(train_x_b, train_adj_b, train_node_flag_b, sigma_list, args, self.device)
                optimizer.zero_grad()
                train_noise_adj_b_chunked = train_noise_adj_b.chunk(len(sigma_list), dim=0)
                train_x_b_chunked = train_x_b.chunk(len(sigma_list), dim=0)
                train_node_flag_sigma = train_node_flag_sigma.chunk(len(sigma_list), dim=0)
                score = []
                masks = []
                for i, sigma in enumerate(sigma_list):
                    mask = generate_mask(train_node_flag_sigma[i])
                    score_batch = model(A=train_noise_adj_b_chunked[i].to(self.device),
                                        node_features=train_x_b_chunked[i].to(self.device), mask=mask.to(self.device),
                                        noiselevel=sigma)   # [bsz, N, N, 1]
                    score.append(score_batch)
                    masks.append(mask)
                graph_batch_sub = tensor2graph(graph, score, mask)
                y_pred, y_exp = self.gnn_pred(graph,  graph_batch_sub, gnn_model, ds=args.dataset_name, task=args.task)
                full_edge_index = gen_full(graph.batch, mask)
                score_b = torch.cat(score, dim=0).squeeze(-1).to(self.device) # [len(sigma_list)*bsz, N, N]
                masktens = torch.cat(masks, dim=0).to(self.device) # [len(sigma_list)*bsz, N, N]
                modif_r = self.sparsity(score, train_adj_b, mask)
                remain_r = self.sparsity(score, train_adj_b, train_adj_b)
                loss_cf, fid_drop, acc_cf = loss_cf_exp(gnn_model, graph, score, y_pred, y_exp, full_edge_index, mask, ds=args.dataset_name, task=args.task)
                loss_dist = loss_func_bce(score_b, train_ori_adj_b, sigma_list, masktens, device=self.device, sparsity_level=args.sparsity_level)
                loss = loss_dist + args.alpha_cf * loss_cf
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_loss_dist.append(loss_dist.item())
                train_loss_cf.append(loss_cf.item())
                train_acc.append(acc_cf)
                train_fid.append(fid_drop)
                train_sparsity.append(modif_r.item())
                train_remain.append(remain_r.item())
                # free up unnecessary memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    gc.collect()
            scheduler.step(epoch)
            mean_train_loss = np.mean(train_losses)
            mean_train_loss_dist = np.mean(train_loss_dist)
            mean_train_loss_cf = np.mean(train_loss_cf)
            mean_train_acc = np.mean(train_acc)
            mean_train_fidelity = np.mean(train_fid)
            mean_train_sparsity = np.mean(train_sparsity)
            mean_remain_rate = np.mean(train_remain)
            print((f'Training Epoch: {epoch} | '
                             f'training loss: {mean_train_loss} | '
                             f'training distribution loss: {mean_train_loss_dist} | '
                             f'training cf loss: {mean_train_loss_cf} | '
                             f'training fidelity drop: {mean_train_fidelity} | '
                             f'training acc: {mean_train_acc} | '
                             f'training average modification: {mean_train_sparsity} | '))

            # free up unnecessary memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()
            # evaluation
            if (epoch + 1) % args.verbose == 0:
                test_losses = []
                test_loss_dist = []
                test_loss_cf = []
                test_acc = []
                test_fid = []
                test_sparsity = []
                test_remain = []
                test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batchsize, shuffle=False)
                model.eval()
                for graph in test_loader:
                    if graph.is_directed() == True:
                        edge_index_temp = graph.edge_index
                        graph.edge_index = to_undirected(edge_index=edge_index_temp)

                    graph.to(self.device)
                    test_adj_b, test_x_b = graph2tensor(graph, device=self.device)
                    test_x_b = test_x_b.to(self.device)
                    test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
                    sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length)) \
                        if noise_list is None else noise_list
                    if isinstance(sigma_list, float):
                        sigma_list = [sigma_list]
                    test_x_b, test_ori_adj_b, test_node_flag_sigma, test_noise_adj_b, noise_diff = \
                        gen_list_of_data_single(test_x_b, test_adj_b, test_node_flag_b, sigma_list, args, self.device)
                    with torch.no_grad():
                        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
                        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
                        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
                        score = []
                        masks = []
                        for i, sigma in enumerate(sigma_list):
                            mask = generate_mask(test_node_flag_sigma[i])
                            score_batch = model(A=test_noise_adj_b_chunked[i].to(self.device),
                                                node_features=test_x_b_chunked[i].to(self.device), mask=mask.to(self.device),
                                                noiselevel=sigma).to(self.device) # [bsz, N, N, 1]
                            masks.append(mask)
                            score.append(score_batch)
                        graph_batch_sub = tensor2graph(graph, score, mask)
                        y_pred, y_exp = self.gnn_pred(graph, graph_batch_sub, gnn_model, ds=args.dataset_name, task=args.task)
                        full_edge_index = gen_full(graph.batch, mask)
                        score_b = torch.cat(score, dim=0).squeeze(-1).to(self.device)
                        masktens = torch.cat(masks, dim=0).to(self.device)
                        modif_r = self.sparsity(score, test_adj_b, mask)
                        reamin_r = self.sparsity(score, test_adj_b, test_adj_b)
                        loss_cf, fid_drop, acc_cf = loss_cf_exp(gnn_model, graph, score, y_pred,y_exp, full_edge_index, mask, ds=args.dataset_name, task=args.task)
                        loss_dist = loss_func_bce(score_b, test_ori_adj_b, sigma_list, masktens, device=self.device, sparsity_level=args.sparsity_level)
                        loss = loss_dist + args.alpha_cf * loss_cf
                        test_losses.append(loss.item())
                        test_loss_dist.append(loss_dist.item())
                        test_loss_cf.append(loss_cf.item())
                        test_acc.append(acc_cf)
                        test_fid.append(fid_drop)
                        test_sparsity.append(modif_r.item())
                        test_remain.append(reamin_r.item())

                        # free up unnecessary memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        else:
                            gc.collect()
                mean_test_loss = np.mean(test_losses)
                mean_test_loss_dist = np.mean(test_loss_dist)
                mean_test_loss_cf = np.mean(test_loss_cf)
                mean_test_acc = np.mean(test_acc)
                mean_test_fid = np.mean(test_fid)
                mean_test_sparsity = np.mean(test_sparsity)
                mean_test_reamin = np.mean(test_remain)

                print((f'Evaluation Epoch: {epoch} | '
                             f'test loss: {mean_test_loss} | '
                             f'test distribution loss: {mean_test_loss_dist} | '
                             f'test cf loss: {mean_test_loss_cf} | '
                             f'test acc: {mean_test_acc} | '
                             f'test fidelity drop: {mean_test_fid} | '
                             f'test average modification: {mean_test_sparsity} | '
                             f'test remain rate: {mean_test_reamin} | '))
                if mean_test_sparsity < best_sparsity:
                    best_sparsity = mean_test_sparsity
                    model_save(args, model) #, mean_train_loss, best_sparsity, mean_test_acc)

    def sparsity(self, score, groundtruth, mask, threshold=0.5):
        '''
        params:
            score: list of [bsz, N, N, 1], list: len(sigma_list),
            groundtruth: [bsz, N, N]
            mask: [bsz, N, N]
        '''
        score_tensor = torch.stack(score, dim=0).squeeze(-1)  # len_sigma_list, bsz, N, N]
        score_tensor = torch.mean(score_tensor, dim=0)  # [bsz, N, N]
        pred_adj = torch.where(torch.sigmoid(score_tensor) > threshold, 1, 0).to(groundtruth.device)
        pred_adj = pred_adj * mask
        groundtruth_ = groundtruth * mask
        adj_diff = torch.abs(groundtruth_ - pred_adj)  # [bsz, N, N]
        num_edge_b = groundtruth_.sum(dim=(1, 2))
        adj_diff_ratio = adj_diff.sum(dim=(1, 2)) / num_edge_b
        ratio_average = torch.mean(adj_diff_ratio)
        return ratio_average

    def _sparsity(self, score, groundtruth, mask, threshold=0.5):
        '''
        params:
            score: list of [bsz, N, N, 1], list: len(sigma_list),
            groundtruth: [bsz, N, N]
            mask: [bsz, N, N]
        '''
        mr_list = []
        for score_tensor in score:  # [bsz, N, N, 1]
            score_tensor = score_tensor.squeeze(-1)  # [ bsz, N, N]
            pred_adj = torch.where(torch.sigmoid(score_tensor) > threshold, 1, 0).to(groundtruth.device)
            pred_adj = pred_adj * mask
            groundtruth_ = groundtruth * mask
            adj_diff = torch.abs(groundtruth_ - pred_adj)  # [bsz, N, N]
            num_edge_b = groundtruth_.sum(dim=(1, 2))
            adj_diff_ratio = adj_diff.sum(dim=(1, 2)) / num_edge_b
            ratio_average = float(torch.mean(adj_diff_ratio).detach().cpu())
            mr_list.append(ratio_average)
        mr = np.mean(mr_list)
        return mr

    def gnn_pred(self, graph_batch, graph_batch_sub, gnn_model, ds, task):
        gnn_model.eval()
        if task == "nc":
            output_prob = gnn_model(graph_batch) #mapping=graph_batch.mapping
            output_prob_sub = gnn_model(graph_batch_sub) #mapping=graph_batch_sub.mapping
        else:
            output_prob = gnn_model(graph_batch)
            output_prob_sub = gnn_model(graph_batch_sub)

        y_pred = output_prob.argmax(dim=-1)
        y_exp = output_prob_sub.argmax(dim=-1)
        return y_pred, y_exp

    def explain_evaluation(self, args, graph):
        model = Powerful(args, self.device).to(self.device)
        exp_dir = f'{args.root}/{args.dataset_name}/'
        model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth"))['model'])
        model.eval()
        graph.to(self.device)
        test_adj_b, test_x_b = graph2tensor(graph, device=self.device) #[bsz, N, N]
        test_x_b = test_x_b.to(self.device)
        test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
        sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length)) \
            if args.noise_list is None else args.noise_list
        if isinstance(sigma_list, float):
            sigma_list = [sigma_list]
        test_x_b, test_ori_adj_b, test_node_flag_sigma, test_noise_adj_b, noise_diff = \
                gen_list_of_data_single(test_x_b, test_adj_b, test_node_flag_b, sigma_list, args, self.device)
        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
        score = []
        masks = []
        for i, sigma in enumerate(sigma_list):
            mask = generate_mask(test_node_flag_sigma[i])
            score_batch = model(A=test_noise_adj_b_chunked[i].to(self.device),
                                    node_features=test_x_b_chunked[i].to(self.device), mask=mask.to(self.device),
                                    noiselevel=sigma).to(self.device)  # [bsz, N, N, 1]
            masks.append(mask)
            score.append(score_batch)
        graph_batch_sub = tensor2graph(graph, score, mask)
        full_edge_index = gen_full(graph.batch, mask)
        modif_r = sparsity(score, test_adj_b, mask)
        score_tensor = torch.stack(score, dim=0).squeeze(-1)  # len_sigma_list, bsz, N, N]
        score_tensor = torch.mean(score_tensor, dim=0).view(-1, 1)  # [bsz*N*N,1]
        mask_bool = mask.bool().view(-1, 1)
        edge_mask_full = score_tensor[mask_bool]
        if args.task == "nc":
            output_prob_cont, output_repr_cont = self.model.get_pred_explain(x=graph.x, edge_index=full_edge_index,
                                                                                edge_mask=edge_mask_full,
                                                                                mapping=graph.mapping)
        else:
            output_prob_cont, output_repr_cont = self.model.get_pred_explain(x=graph.x,
                                                                                edge_index=full_edge_index,
                                                                                edge_mask=edge_mask_full,
                                                                                batch=graph.batch)
        y_ori = graph.y if args.task == "gc" else graph.self_y
        y_exp = output_prob_cont.argmax(dim=-1)
        edge_index_diff = graph_batch_sub.edge_index
        return edge_index_diff, y_ori, y_exp, modif_r


    def explanation_generate(self, args, graph):
        exp_dir = f'{args.root}/{args.dataset_name}/'
        model=torch.load(os.path.join(exp_dir, "best_model.pth")).to(self.device)
        model.eval()
        graph.to(self.device)
        test_adj_b, test_x_b = graph2tensor(graph, device=self.device) #[bsz, N, N]
        test_x_b = test_x_b.to(self.device)
        test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)
        sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length)) \
            if args.noise_list is None else args.noise_list
        if isinstance(sigma_list, float):
            sigma_list = [sigma_list]
        test_x_b, test_ori_adj_b, test_node_flag_sigma, test_noise_adj_b, noise_diff = \
                gen_list_of_data_single(test_x_b, test_adj_b, test_node_flag_b, sigma_list, args, self.device)
        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)
        score = []
        masks = []
        for i, sigma in enumerate(sigma_list):
            mask = generate_mask(test_node_flag_sigma[i])
            score_batch = model(A=test_noise_adj_b_chunked[i].to(self.device),
                                    node_features=test_x_b_chunked[i].to(self.device), mask=mask.to(self.device),
                                    noiselevel=sigma).to(self.device)  # [bsz, N, N, 1]
            masks.append(mask)
            score.append(score_batch)
        graph_batch_sub = tensor2graph(graph, score, test_adj_b) # mask)
        return graph_batch_sub