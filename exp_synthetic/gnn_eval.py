####### Evaluate GNN #######

def get_proba(ypred):
    m = nn.Softmax(dim=1)
    yprob = m(ypred)
    return yprob

def get_labels(ypred):
    ylabels = torch.argmax(ypred, dim=1)
    return ylabels


def gnn_scores(model, data):
    ypred = model(data.x, data.edge_index)
    ylabels = get_labels(ypred).cpu()
    data.y = data.y.cpu()
    
    result_train = {
        "prec": metrics.precision_score(data.y[data.train_mask], ylabels[data.train_mask], average="macro"),
        "recall": metrics.recall_score(data.y[data.train_mask], ylabels[data.train_mask], average="macro"),
        "acc": metrics.accuracy_score(data.y[data.train_mask], ylabels[data.train_mask])
        #"conf_mat": metrics.confusion_matrix(data.y[data.train_mask], ylabels[data.train_mask]),
    }

    result_test = {
        "prec": metrics.precision_score(data.y[data.test_mask], ylabels[data.test_mask], average="macro"),
        "recall": metrics.recall_score(data.y[data.test_mask], ylabels[data.test_mask], average="macro"),
        "acc": metrics.accuracy_score(data.y[data.test_mask], ylabels[data.test_mask])#,
        #"conf_mat": metrics.confusion_matrix(data.y[data.test_mask], ylabels[data.test_mask]),
    }
    return result_train, result_test

