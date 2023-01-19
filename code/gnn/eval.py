import numpy as np
import torch
from dataset.mutag_utils import prepare_data
from sklearn import metrics
from torch.autograd import Variable
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from tqdm import tqdm
from utils.gen_utils import from_adj_to_edge_index, get_labels
from utils.graph_utils import get_edge_index_batch


def gnn_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def gnn_scores_nc(model, data, args, device):
    model.eval()

    if data.num_nodes > args.sample_size:
        cluster_data = ClusterData(
            data, num_parts=data.x.size(0) // args.sample_size, recursive=False
        )
        data_loader = ClusterLoader(
            cluster_data, batch_size=1, shuffle=True
        )  # , num_workers=args.num_workers)
        data = next(iter(data_loader))

    ypred = (
        model(data.x, data.edge_index, edge_weight=data.edge_weight)
        .cpu()
        .detach()
        .numpy()
    )
    ylabels = get_labels(ypred)
    data.y = data.y.cpu()

    data.train_mask = data.train_mask.cpu()
    data.test_mask = data.test_mask.cpu()
    print("lenght of test mask", len(data.test_mask[data.test_mask == True]))

    result_train = {
        "prec": metrics.precision_score(
            data.y[data.train_mask], ylabels[data.train_mask], average="macro"
        ),
        "recall": metrics.recall_score(
            data.y[data.train_mask], ylabels[data.train_mask], average="macro"
        ),
        "f1-score": metrics.f1_score(
            data.y[data.train_mask], ylabels[data.train_mask], average="macro"
        ),
        "acc": metrics.accuracy_score(
            data.y[data.train_mask], ylabels[data.train_mask]
        ),
    }

    result_test = {
        "prec": metrics.precision_score(
            data.y[data.test_mask], ylabels[data.test_mask], average="macro"
        ),
        "recall": metrics.recall_score(
            data.y[data.test_mask], ylabels[data.test_mask], average="macro"
        ),
        "f1-score": metrics.f1_score(
            data.y[data.test_mask], ylabels[data.test_mask], average="macro"
        ),
        "acc": metrics.accuracy_score(data.y[data.test_mask], ylabels[data.test_mask]),
    }
    return result_train, result_test
