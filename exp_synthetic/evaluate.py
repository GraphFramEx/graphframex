import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from synthetic_structsim import house
from explainer import node_attr_to_edge
from gengroundtruth import get_ground_truth
from sklearn import metrics
import torch
from code.utils.gen_utils import list_to_dict, get_subgraph
from gnn_eval import get_proba

def topk_edges_directed(edge_mask, edge_index, num_top_edges):
    indices = (-edge_mask).argsort()
    top = np.array([], dtype='int')
    i = 0
    list_edges = np.sort(edge_index.cpu().T, axis=1)
    while len(top)<num_top_edges:
        subset = indices[num_top_edges*i:num_top_edges*(i+1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]])
        i+=1
    return top[:num_top_edges]

    

def normalize_mask(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def normalize_all_masks(masks):
    for i in range(len(masks)):
        masks[i] = normalize_mask(masks[i])
    return masks

def clean_masks(masks):
    for i in range(len(masks)):
        masks[i] = np.nan_to_num(masks[i], copy=True, nan=0.0, posinf=10, neginf=-10)
        masks[i] = np.clip(masks[i], -10, 10)
    return masks


def get_sparsity(masks):
    sparsity = 0
    for i in range(len(masks)):
        sparsity += 1.0 - (masks[i]!= 0).sum() / len(masks[i])
    return sparsity/len(masks)

def get_size(masks):
    size = 0
    for i in range(len(masks)):
        size += len(masks[i][masks[i]>0])
    return size/len(masks)
# Edge_masks are normalized; we then select only the edges for which the mask value > threshold
#transform edge_mask:
# normalisation
# sparsity
# hard or soft

def transform_mask(masks, args):
    new_masks = []
    for mask in masks:
        if args.topk >= 0:
            unimportant_indices = (-mask).argsort()[args.topk:]
            mask[unimportant_indices] = 0
        if args.sparsity>=0:
            mask = control_sparsity(mask, args.sparsity)
        if args.threshold>=0:
            mask = np.where(mask>args.threshold, mask, 0)
        new_masks.append(mask)
    return(new_masks)

def mask_to_shape(mask, edge_index, num_top_edges):
    indices = topk_edges_directed(mask, edge_index, num_top_edges)
    new_mask = np.zeros(len(mask))
    new_mask[indices] = 1
    return new_mask

def control_sparsity(mask, sparsity):
    r"""
        :param edge_mask: mask that need to transform
        :param sparsity: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
    """
    mask_len = len(mask)
    split_point = int((1 - sparsity) * mask_len)
    unimportant_indices = (-mask).argsort()[split_point:]
    mask[unimportant_indices] = 0
    return mask


##### Accuracy #####
def get_explanation(data, edge_mask, **kwargs): 
    if kwargs['top']:
        #indices = (-edge_mask).argsort()[:kwargs['num_top_edges']]
        edge_mask = mask_to_shape(edge_mask, data.edge_index, kwargs['num_top_edges'])
        indices = np.where(edge_mask>0)[0]
    else:
        indices = np.where(edge_mask>0)[0]
    
    explanation = data.edge_index[:, indices]
    G_expl = nx.Graph()
    G_expl.add_nodes_from(np.unique(explanation.cpu()))
    for i, (u, v) in enumerate(explanation.t().tolist()):
        G_expl.add_edge(u, v)
    return (G_expl)




def get_scores(G1, G2, **kwargs):
    G1, G2 = G1.to_undirected(), G2.to_undirected()
    g_int = nx.intersection(G1, G2)
    g_int.remove_nodes_from(list(nx.isolates(g_int)))
    
    n_tp = g_int.number_of_edges()
    n_fp = len(G1.edges() - g_int.edges())
    n_fn = len(G2.edges() - g_int.edges())
    
    if n_tp == 0:
        precision, recall = 0, 0
        f1_score = 0
    else:
        precision = n_tp / (n_tp + n_fp)
        recall = n_tp / (n_tp + n_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    if kwargs['top']:
        ged = -1 #nx.graph_edit_distance(G1, G2)
    else: #Too long to compute
        ged = -1
    return recall, precision, f1_score, ged

def get_accuracy(data, edge_mask, node_idx, args, **kwargs):
    G_true, role, true_edge_mask = get_ground_truth(node_idx, data, args)
    # nx.draw(G_true, with_labels=True, font_weight='bold')
    G_expl = get_explanation(data, edge_mask, **kwargs)
    # plt.figure()
    # nx.draw_networkx(G_expl, with_labels=True, font_weight='bold')
    # plt.show()
    # plt.clf()
    recall, precision, f1_score, ged = get_scores(G_expl, G_true, **kwargs)
    #fpr, tpr, thresholds = metrics.roc_curve(true_edge_mask, edge_mask, pos_label=1)
    #auc = metrics.auc(fpr, tpr)
    auc = -1
    return {'recall': recall, 'precision': precision, 'f1_score': f1_score, 'ged': ged, 'auc': auc}


def eval_accuracy(data, edge_masks, list_node_idx, args, **kwargs):
    n_test = len(list_node_idx)
    scores = []

    for i in range(n_test):
        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]
        entry = get_accuracy(data, edge_mask, node_idx, args, **kwargs)
        scores.append(entry)

    scores = list_to_dict(scores)
    accuracy_scores = {k: np.mean(v) for k, v in scores.items()}
    return accuracy_scores


##### Fidelity #####
def eval_related_pred_subgraph(model, data, edge_masks, list_node_idx, device, **kwargs):
    related_preds = []
    data = data.to(device)
    n_test = len(list_node_idx)

    for i in range(n_test):

        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]
        mask_sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)
        
        sub_x, sub_edge_index, mapping, hard_edge_mask, subset, _ = get_subgraph(node_idx, data.x, data.edge_index, 2)

        ori_ypred = model(sub_x.to(device), sub_edge_index.to(device))
        ori_yprob = get_proba(ori_ypred)

        sub_masked_x, sub_maskout_x = sub_x, sub_x

        sub_edge_mask = edge_mask[hard_edge_mask.cpu()]

        indices = np.where(sub_edge_mask > 0)[0]
        indices_inv = [i for i in range(len(sub_edge_mask)) if i not in indices]

        sub_masked_edge_index = sub_edge_index[:, indices]
        sub_maskout_edge_index = sub_edge_index[:, indices_inv]

        def masking(indices):
            masked_edge_index = data.edge_index[:, indices]
            masked_nodes = torch.LongTensor(np.unique(masked_edge_index))
            if node_idx not in masked_nodes:
                masked_nodes = torch.cat([torch.LongTensor([node_idx]), masked_nodes])

            # Function to be vectorized
            def map_func(val, dictionary):
                return dictionary[val] if val in dictionary else val

            vfunc = np.vectorize(map_func)
            d = {v.item(): k for k, v in enumerate(masked_nodes)}
            sub_masked_edge_index = torch.LongTensor(vfunc(masked_edge_index, d))
            masked_node_idx = int(vfunc(node_idx, d))
            sub_masked_x = torch.FloatTensor([[1]] * len(masked_nodes)).to(device)
            return sub_masked_x, sub_masked_edge_index, masked_node_idx

        # sub_masked_x, sub_masked_edge_index, masked_node_idx = masking(indices)
        # sub_maskout_x, sub_maskout_edge_index, maskout_node_idx = masking(indices_inv)

        if kwargs['hard_mask']:
            masked_ypred = model(sub_masked_x.to(device), sub_masked_edge_index.to(device))
            masked_yprob = get_proba(masked_ypred)

            maskout_ypred = model(sub_maskout_x.to(device), sub_maskout_edge_index.to(device))
            maskout_yprob = get_proba(maskout_ypred)

        else:
            masked_ypred = model(sub_masked_x.to(device), sub_masked_edge_index.to(device), edge_weight=edge_mask[indices].to(device))
            masked_yprob = get_proba(masked_ypred)

            maskout_ypred = model(sub_maskout_x.to(device), sub_maskout_edge_index.to(device), edge_weight=(1 - edge_mask)[indices_inv].to(device))
            maskout_yprob = get_proba(maskout_ypred)

        ori_probs = ori_yprob[mapping.item()].detach().cpu().numpy()
        masked_probs = masked_yprob[mapping.item()].detach().cpu().numpy()
        maskout_probs = maskout_yprob[mapping.item()].detach().cpu().numpy()

        true_label = data.y[node_idx].cpu().numpy()
        pred_label = np.argmax(ori_probs)
        # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

        related_preds.append({'node_idx': node_idx,
                              'masked': masked_probs,
                              'maskout': maskout_probs,
                              'origin': ori_probs,
                              'sparsity': mask_sparsity,
                              'true_label': true_label,
                              'pred_label': pred_label})

    related_preds = list_to_dict(related_preds)
    return related_preds


def eval_related_pred(model, data, edge_masks, list_node_idx, device):
    related_preds = []
    data = data.to(device)
    ori_ypred = model(data.x, data.edge_index)
    ori_yprob = get_proba(ori_ypred)

    n_test = len(list_node_idx)

    for i in range(n_test):
        edge_mask = torch.Tensor(edge_masks[i])
        node_idx = list_node_idx[i]
        mask_sparsity = 1.0 - (edge_mask != 0).sum() / edge_mask.size(0)
        
        indices = np.where(edge_mask > 0)[0]
        indices_inv = [i for i in range(len(edge_mask)) if i not in indices]

        masked_edge_index = data.edge_index[:, indices].to(device)
        maskout_edge_index = data.edge_index[:, indices_inv].to(device)

        masked_ypred = model(data.x, masked_edge_index)
        masked_yprob = get_proba(masked_ypred)

        maskout_ypred = model(data.x, maskout_edge_index)
        maskout_yprob = get_proba(maskout_ypred)

        ori_probs = ori_yprob[node_idx].detach().cpu().numpy()
        masked_probs = masked_yprob[node_idx].detach().cpu().numpy()
        maskout_probs = maskout_yprob[node_idx].detach().cpu().numpy()

        true_label = data.y[node_idx].cpu().numpy()
        pred_label = np.argmax(ori_probs)
        # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

        related_preds.append({'node_idx': node_idx,
                              'masked': masked_probs,
                              'maskout': maskout_probs,
                              'origin': ori_probs,
                              'mask_sparsity': mask_sparsity,
                              'expl_edges': (edge_mask != 0).sum(),
                              'true_label': true_label,
                              'pred_label': pred_label})

    related_preds = list_to_dict(related_preds)
    return related_preds


def fidelity_acc(related_preds):
    labels = related_preds['true_label']
    ori_labels = np.argmax(related_preds['origin'], axis=1)
    unimportant_labels = np.argmax(related_preds['maskout'], axis=1)
    p_1 = np.array(ori_labels == labels).astype(int)
    p_2 = np.array(unimportant_labels == labels).astype(int)
    drop_probability = p_1 - p_2
    return drop_probability.mean().item()


def fidelity_acc_inv(related_preds):
    labels = related_preds['true_label']
    ori_labels = np.argmax(related_preds['origin'], axis=1)
    important_labels = np.argmax(related_preds['masked'], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = p_1 - p_2
    return drop_probability.mean().item()


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_prob(related_preds):
    labels = related_preds['true_label']
    ori_probs = np.choose(labels, related_preds['origin'].T)
    unimportant_probs = np.choose(labels, related_preds['maskout'].T)
    drop_probability = ori_probs - unimportant_probs
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_prob_inv(related_preds):
    labels = related_preds['true_label']
    ori_probs = np.choose(labels, related_preds['origin'].T)
    important_probs = np.choose(labels, related_preds['masked'].T)
    drop_probability = ori_probs - important_probs
    return drop_probability.mean().item()


def eval_fidelity(related_preds):
    fidelity_scores = {
        "fidelity_acc+": fidelity_acc(related_preds),
        "fidelity_acc-": fidelity_acc_inv(related_preds),
        "fidelity_prob+": fidelity_prob(related_preds),
        "fidelity_prob-": fidelity_prob_inv(related_preds),
    }
    return fidelity_scores

