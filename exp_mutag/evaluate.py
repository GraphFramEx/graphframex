from sklearn import metrics


def gnn_scores(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        if args.gpu:
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float()).cuda()
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()
        else:
            adj = Variable(data["adj"].float(), requires_grad=False)
            h0 = Variable(data["feats"].float())
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            )
            
        edge_index = []
        for a in adj:
            edge_index.append(from_adj_to_edge_index(a))

        ypred = model(h0, edge_index, batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {
        "prec": metrics.precision_score(labels, preds, average="macro"),
        "recall": metrics.recall_score(labels, preds, average="macro"),
        "acc": metrics.accuracy_score(labels, preds),
    }
    print(name, " accuracy:", result["acc"])
    return result


import torch
from torch.nn import Softmax
from scipy.special import softmax
##### Fidelity #####

def gnn_preds(model, dataset, edge_index_set, max_num_examples=None):
    model.eval()
    labels = []
    pred_labels = []
    ypreds = []
    for batch_idx, data in enumerate(dataset):
        if args.gpu:
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float()).cuda()
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            ).cuda()
        else:
            adj = Variable(data["adj"].float(), requires_grad=False)
            h0 = Variable(data["feats"].float())
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(
                data["assign_feats"].float(), requires_grad=False
            )
        
        ypred = model(h0, edge_index_set[batch_idx], batch_num_nodes, assign_x=assign_input)
        _, indices = torch.max(ypred, 1)
        pred_labels.append(indices.cpu().data.numpy())
        ypreds.append(ypred.cpu().data.numpy())

        
        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    labels = np.hstack(labels)
    pred_labels = np.hstack(pred_labels)
    ypreds = np.concatenate(ypreds)
    return(ypreds)
    
def get_true_labels(dataset):
    labels = []
    for batch_idx, data in enumerate(dataset):
        labels.append(data["label"].long().numpy())
    labels = np.hstack(labels)
    return labels
    
def get_proba(ypred):
    yprob = softmax(ypred, axis=1)
    return yprob

def get_labels(ypred):
    ylabels = torch.argmax(ypred, dim=1)
    return ylabels

def list_to_dict(preds):
    preds_dict=pd.DataFrame(preds).to_dict('list')
    for key in preds_dict.keys():
        preds_dict[key] = preds_dict[key][0]
    return(preds_dict)

def eval_related_pred_batch(model, dataset, edge_index_set, edge_masks_set, device):
    n_graphs = len(np.hstack(edge_masks_set))
    n_bs = len(edge_masks_set)
    
    mask_sparsity = 0
    expl_edges = 0
    for edge_mask in np.hstack(edge_masks_set):
        mask_sparsity += 1.0 - (edge_mask != 0).sum() / len(edge_mask)
        expl_edges += (edge_mask != 0).sum()
    mask_sparsity /= n_graphs
    expl_edges /= n_graphs
    
    #related_preds = []
    
    ori_ypred = eval_gnn(model, dataset, edge_index_set)
    ori_yprob = get_proba(ori_ypred)

    masked_edge_index_set, maskout_edge_index_set = compute_masked_edges(edge_masks_set, edge_index_set)

    masked_ypred = eval_gnn(model, dataset, masked_edge_index_set)
    masked_yprob = get_proba(masked_ypred)

    maskout_ypred = eval_gnn(model, dataset, maskout_edge_index_set)
    maskout_yprob = get_proba(maskout_ypred)

    #true_label = [data["label"].long().numpy() for batch_idx, data in enumerate(dataset)]
    #pred_label = get_labels(ori_ypred)
    #print(true_label)
    #print(pred_label)
    # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."

    related_preds = {'masked': masked_yprob,
                          'maskout': maskout_yprob,
                          'origin': ori_yprob,
                          'mask_sparsity': mask_sparsity,
                          'expl_edges': expl_edges, 
                        'true_label': get_true_labels(dataset)
                    }
    #related_preds = list_to_dict(related_preds)
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


