import numpy as np
import torch
from scipy.special import softmax
from sklearn import metrics
from torch.autograd import Variable
from utils.gen_utils import from_adj_to_edge_index, get_labels


def gnn_scores_nc(model, data):
    ypred = model(data.x, data.edge_index)
    ylabels = get_labels(ypred).cpu()
    data.y = data.y.cpu()

    result_train = {
        "prec": metrics.precision_score(data.y[data.train_mask], ylabels[data.train_mask], average="macro"),
        "recall": metrics.recall_score(data.y[data.train_mask], ylabels[data.train_mask], average="macro"),
        "acc": metrics.accuracy_score(data.y[data.train_mask], ylabels[data.train_mask]),
    }

    result_test = {
        "prec": metrics.precision_score(data.y[data.test_mask], ylabels[data.test_mask], average="macro"),
        "recall": metrics.recall_score(data.y[data.test_mask], ylabels[data.test_mask], average="macro"),
        "acc": metrics.accuracy_score(data.y[data.test_mask], ylabels[data.test_mask]),  # ,
    }
    return result_train, result_test


def gnn_scores_gc(dataset, model, args, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        if args.gpu:
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float()).cuda()
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(data["assign_feats"].float(), requires_grad=False).cuda()
        else:
            adj = Variable(data["adj"].float(), requires_grad=False)
            h0 = Variable(data["feats"].float())
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(data["assign_feats"].float(), requires_grad=False)

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


def gnn_preds_gc(model, dataset, edge_index_set, args, max_num_examples=None):
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
            assign_input = Variable(data["assign_feats"].float(), requires_grad=False).cuda()
        else:
            adj = Variable(data["adj"].float(), requires_grad=False)
            h0 = Variable(data["feats"].float())
            labels.append(data["label"].long().numpy())
            batch_num_nodes = data["num_nodes"].int().numpy()
            assign_input = Variable(data["assign_feats"].float(), requires_grad=False)

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
    return ypreds
