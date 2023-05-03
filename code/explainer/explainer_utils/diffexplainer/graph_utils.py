import numpy as np
import torch
from torch_geometric.utils import to_dense_adj, degree

do_check_adjs_symmetry = False


def mask_adjs(adjs, node_flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param node_flags: B x N
    :return:
    """
    if len(adjs.shape) == 4:
        node_flags = node_flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * node_flags.unsqueeze(-1)
    adjs = adjs * node_flags.unsqueeze(-2)
    return adjs


def get_corrupt_k(min_k=0, max_k=None, p=0.5):
    ret = np.random.geometric(p) + min_k - 1
    if max_k is not None:
        ret = min(ret, max_k)
    # print(ret, end=' ')
    return ret


def remove_self_loop_if_exists(adjs):
    return (adjs - torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)).clamp(min=0.0)


def add_self_loop_if_not_exists(adjs):
    if len(adjs.shape) == 4:
        return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).unsqueeze(0).to(adjs.device)
    return adjs + torch.eye(adjs.size()[-1]).unsqueeze(0).to(adjs.device)


def toggle_edge_np(adj, count=1):
    """
    uniformly toggle `count` edges of the graph, suppose that the vertex number is fixed
    :param adj: N x N
    :param count: int
    :return: new adjs and node_flags
    """
    count = min(count, adj.shape[-1])
    x = np.random.choice(adj.shape[0], count)
    y = np.random.choice(adj.shape[1], count)
    change = 1. - adj[x, y]
    adj[x, y] = change
    adj[y, x] = change
    return adj


def check_adjs_symmetry(adjs):
    if not do_check_adjs_symmetry:
        return
    tr_adjs = adjs.transpose(-1, -2)
    assert (adjs - tr_adjs).abs().sum([0, 1, 2]) < 1e-2

def gen_list_of_data_single(train_x_b, train_adj_b, train_node_flag_b, sigma_list, args, device):
    """
    :param train_x_b: [batch_size, N, F_in], batch of feature vectors of nodes
    :param train_adj_b: [batch_size, N, N], batch of original adjacency matrices
    :param train_node_flag_b: [batch_size, N], the flags for the existence of nodes
    :param sigma_list: list of noise levels
    :returns:
        train_x_b: [len(sigma_list) * batch_size, N, F_in], batch of feature vectors of nodes
        train_ori_adj_b: [len(sigma_list) * batch_size, N, N], batch of original adjacency matrix (considered as the groundtruth)
        train_node_flag_b: [len(sigma_list) * batch_size, N], the flags for the existence of nodes
        train_noise_adj_b: [len(sigma_list) * batch_size, N, N], batch of noisy adjacency matrices
        noise_list: [len(sigma_list) * batch_size, N, N], the noise added to graph
    """
    assert isinstance(sigma_list, list)
    train_noise_adj_b_list = []
    noise_list = []
    for i, sigma_i in enumerate(sigma_list):
        train_noise_adj_b, true_noise = discretenoise_single(
            train_adj_b, node_flags=train_node_flag_b, sigma=sigma_i, device=device)

        train_noise_adj_b_list.append(train_noise_adj_b)
        noise_list.append(true_noise)

    train_noise_adj_b = torch.cat(train_noise_adj_b_list, dim=0).to(device)
    noise_list = torch.cat(noise_list, dim=0).to(device)
    train_x_b = train_x_b.repeat(len(sigma_list), 1, 1)
    train_ori_adj_b = train_adj_b.repeat(len(sigma_list), 1, 1)
    train_node_flag_sigma = train_node_flag_b.repeat(len(sigma_list), 1)
    return train_x_b, train_ori_adj_b, train_node_flag_sigma, train_noise_adj_b, noise_list

def discretenoise_single(train_adj_b, node_flags, sigma, device):

    train_adj_b = train_adj_b.to(device)
    ##if Aij=1 then chances for being 1 later is 1-sigma so chance of changing is sigma
    bernoulli_adj = torch.where(train_adj_b > 1 / 2, torch.full_like(train_adj_b, sigma).to(device),
                                torch.full_like(train_adj_b, sigma).to(device))

    noise_upper = torch.bernoulli(bernoulli_adj).triu(diagonal=1).to(device)
    noise_lower = noise_upper.transpose(-1, -2)
    train_adj = torch.abs(-train_adj_b + noise_upper + noise_lower)
    noisediff = noise_upper + noise_lower
    train_adj = mask_adjs(train_adj, node_flags)
    noisediff = mask_adjs(noisediff, node_flags)
    return train_adj, noisediff


def generate_mask(node_flags):
    '''
    params:
        node_flages： [bsz, N]
    returns:
        groundtruth: [bsz, N, N]
    '''
    flag2 = node_flags.unsqueeze(1) #[bsz,1,N]
    flag1 = node_flags.unsqueeze(-1)  #[bsz,N,1]
    mask_matrix = torch.bmm(flag1, flag2)  #[bsz, N, N]
    groundtruth = torch.where(mask_matrix > 0.9, 1, 0).to(node_flags.device)
    return groundtruth


def graph2tensor(graph, device):
    '''
    Args:
        graph: graph batch

    Returns:
        adj: [bsz, N, N]
        x: [bsz, N, C]
    '''

    bsz = graph.num_graphs
    edge_index = graph.edge_index  #[2, E_total]
    adj = to_dense_adj(edge_index, batch=graph.batch)  #[bsz, max_num_node, max_num_node]
    max_num_node = adj.size(-1)
    node_features = graph.x        #[N_total, C]
    feature_dim = node_features.size(-1)
    node_sizes = degree(graph.batch, dtype=torch.long).tolist()
    x_split = node_features.split(node_sizes, dim=0)  #list of tensor
    x_tensor = torch.empty((bsz, max_num_node, feature_dim)).to(device)
    assert len(x_split) == bsz
    for i in range(bsz):
        Gi_x = x_split[i]
        num_node = Gi_x.size(0)
        zero_tensor = torch.zeros((max_num_node - num_node, feature_dim)).to(device)
        Gi_x = torch.cat((Gi_x,zero_tensor), dim=0)
        assert Gi_x.size(0) == max_num_node
        x_tensor[i] = Gi_x
    return adj, x_tensor

def tensor2graph(graph_batch, score, mask_adj, threshold=0.5):
    '''
    Args:
        graph_batch: graph batch
        score: list of [bsz, N, N, 1], list: len(sigma_list),
        mask_adj: [bsz, N, N]
    '''
    score_tensor = torch.stack(score, dim=0).squeeze(-1)  #len_sigma_list, bsz, N, N]
    score_tensor = torch.mean(score_tensor, dim=0)  #[bsz, N, N]
    bsz = score_tensor.size(0)
    pred_adj = torch.where(torch.sigmoid(score_tensor) > threshold, 1, 0).to(score_tensor.device)
    pred_adj = pred_adj * mask_adj
    edge_indices = pred_adj.nonzero().t()
    node_sizes = degree(graph_batch.batch, dtype=torch.long).detach().cpu().numpy()  #list of node numbers
    sum_list = torch.tensor([node_sizes[:i].sum() for i in range(bsz)]).to(score_tensor.device)
    
    batch = sum_list[edge_indices[0]]
    row = batch + edge_indices[1]
    col = batch + edge_indices[2]
    edge_index = torch.stack([row, col], dim=0)

    graph_batch_sub = graph_batch.clone()
    graph_batch_sub.edge_index = edge_index
    edge_attr = graph_batch.edge_attr
    graph_batch_sub.edge_attr = torch.ones((edge_index.size(1), edge_attr.size(1))).to(edge_attr.device)
    graph_batch_sub.edge_weight = torch.sigmoid(score_tensor[edge_indices[0], edge_indices[1], edge_indices[2]])

    return graph_batch_sub


def gen_full(batch, mask):
    '''
    Args:
        batch: graph.batch
        mask:[bsz, N, N]
    '''
    bsz = mask.size(0)
    node_sizes = degree(batch, dtype=torch.long).detach().cpu().numpy()  # list of node numbers
    sum_list = torch.tensor([node_sizes[:i].sum() for i in range(bsz)]).to(mask.device)
    edge_indices = mask.nonzero().t()
    batch = sum_list[edge_indices[0]]
    row = batch + edge_indices[1]
    col = batch + edge_indices[2]
    edge_index = torch.stack([row, col], dim=0)
    return edge_index