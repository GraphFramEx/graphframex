import numpy as np
import torch
from utils.graph_utils import get_edge_index_set
from dataset.mutag_utils import prepare_data
from sklearn import metrics
from torch.autograd import Variable
from utils.gen_utils import from_adj_to_edge_index, get_labels


def gnn_scores_nc(model, data):
    ypred = model(data.x, data.edge_index).cpu().detach().numpy()
    ylabels = get_labels(ypred)
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


def gnn_scores_gc(model, dataset, args, device):
    train_dataset, val_dataset, test_dataset, max_num_nodes, feat_dim, assign_feat_dim = prepare_data(dataset, args)
    model.eval()
    result_train = evaluate_gc(model, train_dataset, args, device, name="Train")
    result_test = evaluate_gc(model, test_dataset, args, device, name="Test")
    return result_train, result_test


def gnn_preds_gc(model, dataset, edge_index, args, device, max_num_examples=None):
    model.eval()
    pred_labels = []
    ypreds = []
    for i in range(len(dataset)):
        data = dataset[i]
        h0 = Variable(torch.Tensor(data["feats"])).to(device)
        edges = Variable(torch.LongTensor(edge_index[i])).to(device)
        ypred = model(h0, edges)
        _, indices = torch.max(ypred, 1)
        pred_labels.append(indices.cpu().data.numpy())
        ypreds.append(ypred.cpu().data.numpy())
    ypreds = np.concatenate(ypreds)
    return ypreds


def evaluate_gc(model, dataset, args, device, name="Validation", max_num_examples=None):
    model.eval()

    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data["adj"].float(), requires_grad=False).to(device)
        h0 = Variable(data["feats"].float()).to(device)
        labels.append(data["label"].long().numpy())
        batch_num_nodes = data["num_nodes"].int().numpy()
        assign_input = Variable(data["assign_feats"].float(), requires_grad=False).to(device)

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
