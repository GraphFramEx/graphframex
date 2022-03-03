import json
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import utils.io_utils
import utils.math_utils
from dataset.mutag_utils import prepare_data
from torch.autograd import Variable
from torch_geometric.loader import ClusterData, ClusterLoader
from utils.gen_utils import from_adj_to_edge_index

from gnn.eval import *


####### GNN Training #######
def train_node_classification(model, data, device, args):

    # if eval(args.batch):
    if data.num_nodes > args.sample_size:

        cluster_data = ClusterData(data, num_parts=data.x.size(0) // 2000, recursive=False)
        train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_err = []
        train_err = []

        for epoch in range(args.num_epochs):
            total_nodes = total_loss = 0
            model.train()
            for batch_idx in range(len(train_loader)):
                batch = next(iter(train_loader))
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = model.loss(out[batch.train_mask], batch.y[batch.train_mask])

                loss.backward()
                optimizer.step()

                train_nodes = batch.train_mask.sum().item()
                total_loss += loss.item() * train_nodes
                total_nodes += train_nodes

            loss = total_loss / total_nodes

            if epoch % 10 == 0:
                val_loss = model.loss(out[batch.val_mask], batch.y[batch.val_mask]).item() / batch.val_mask.sum().item()
                val_err.append(val_loss)
                train_err.append(loss)
                print("__logs:" + json.dumps({"val_err": round(val_loss, 4), "train_err": round(loss, 4)}))

    else:

        data = data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        val_err = []
        train_err = []
        model.train()
        for epoch in range(args.num_epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = model.loss(out[data.train_mask], data.y[data.train_mask])
            val_loss = model.loss(out[data.val_mask], data.y[data.val_mask])

            if epoch % 10 == 0:
                val_err.append(val_loss.item())
                train_err.append(loss.item())
                print(
                    "__logs:" + json.dumps({"val_err": round(val_loss.item(), 4), "train_err": round(loss.item(), 4)})
                )

            loss.backward()
            optimizer.step()

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(range(args.num_epochs // 10), train_err)
    plt.plot(range(args.num_epochs // 10), val_err, "-", lw=1)
    plt.legend(["train", "val"])
    plt.savefig(utils.io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")


def train_graph_classification(model, data, device, args, mask_nodes=True):
    """Train GNN model.

    Args:
        model ([type]): GNN model
        train_dataset ([type]): [description]
        val_dataset ([type]): [description]
        test_dataset ([type]): [description]
        device ([type]): [description]
        args ([type]): [description]
        same_feat (bool, optional): [description]. Defaults to True.
        writer ([type], optional): [description]. Defaults to None.
        mask_nodes (bool, optional): [description]. Defaults to True.
    """
    train_dataset, val_dataset, test_dataset, max_num_nodes, feat_dim, assign_feat_dim = prepare_data(data, args)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    iter = 0
    best_val_result = {"epoch": 0, "loss": 0, "acc": 0}
    test_result = {"epoch": 0, "loss": 0, "acc": 0}
    train_accs = []
    train_epochs = []
    best_val_accs = []
    best_val_epochs = []
    test_accs = []
    test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        begin_time = time.time()
        avg_loss = 0.0
        model.train()
        predictions = []
        print("Epoch: ", epoch)
        for batch_idx, data in enumerate(train_dataset):
            model.zero_grad()
            if batch_idx == 0:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = prev_adjs
                all_feats = prev_feats
                all_labels = prev_labels
            elif batch_idx < 20:
                prev_adjs = data["adj"]
                prev_feats = data["feats"]
                prev_labels = data["label"]
                all_adjs = torch.cat((all_adjs, prev_adjs), dim=0)
                all_feats = torch.cat((all_feats, prev_feats), dim=0)
                all_labels = torch.cat((all_labels, prev_labels), dim=0)

            adj = Variable(data["adj"].float(), requires_grad=False).to(device)
            h0 = Variable(data["feats"].float(), requires_grad=False).to(device)
            label = Variable(data["label"].long()).to(device)
            batch_num_nodes = data["num_nodes"].int().numpy() if mask_nodes else None
            assign_input = Variable(data["assign_feats"].float(), requires_grad=False).to(device)

            edge_index = []
            for a in adj:
                edges, _ = from_adj_to_edge_index(a)
                edge_index.append(edges)

            ypred = model(h0, edge_index, batch_num_nodes, assign_x=assign_input)
            # ypred, att_adj = model(h0, adj, batch_num_nodes, assign_x=assign_input)
            if batch_idx < 5:
                predictions += ypred.cpu().detach().numpy().tolist()

            if not args.method == "soft-assign" or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            iter += 1
            avg_loss += loss

        avg_loss /= batch_idx + 1
        elapsed = time.time() - begin_time
        print("Avg loss: ", avg_loss, "; epoch time: ", elapsed)

        result = evaluate_gc(model, train_dataset, args, device, name="Train", max_num_examples=100)
        train_accs.append(result["acc"])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate_gc(model, val_dataset, args, device, name="Validation")
            val_accs.append(val_result["acc"])
        if val_result["acc"] > best_val_result["acc"] - 1e-7:
            best_val_result["acc"] = val_result["acc"]
            best_val_result["epoch"] = epoch
            best_val_result["loss"] = avg_loss
        if test_dataset is not None:
            test_result = evaluate_gc(model, test_dataset, args, device, name="Test")
            test_result["epoch"] = epoch
        print("Best val result: ", best_val_result)
        best_val_epochs.append(best_val_result["epoch"])
        best_val_accs.append(best_val_result["acc"])
        if test_dataset is not None:
            print("Test result: ", test_result)
            test_epochs.append(test_result["epoch"])
            test_accs.append(test_result["acc"])

    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    plt.figure()
    plt.plot(train_epochs, utils.math_utils.exp_moving_avg(train_accs, 0.85), "-", lw=1)
    if test_dataset is not None:
        plt.plot(best_val_epochs, best_val_accs, "bo", test_epochs, test_accs, "go")
        plt.legend(["train", "val", "test"])
    else:
        plt.plot(best_val_epochs, best_val_accs, "bo")
        plt.legend(["train", "val"])
    plt.savefig(utils.io_utils.gen_train_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")

    print(all_adjs.shape, all_feats.shape, all_labels.shape)
    return
