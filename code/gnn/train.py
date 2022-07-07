import json
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import utils.io_utils
import utils.math_utils
from torch.autograd import Variable
from torch_geometric.loader import ClusterData, ClusterLoader

from gnn.eval import *


####### GNN Training #######


def train_real_nc(model, data, device, args):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data.to(device)

    # Train model
    t_total = time.time()
    for epoch in range(args.num_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        # output = model(features, data.edge_index, data.edge_weight)

        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])

        acc_train = gnn_accuracy(output[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()

        model.eval()
        output = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        
        loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        # loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = gnn_accuracy(output[data.val_mask], data.y[data.val_mask])
        print(
            "Epoch: {:04d}".format(epoch + 1),
            "loss_train: {:.4f}".format(loss_train.item()),
            "acc_train: {:.4f}".format(acc_train.item()),
            "loss_val: {:.4f}".format(loss_val.item()),
            "acc_val: {:.4f}".format(acc_val.item()),
            "time: {:.4f}s".format(time.time() - t),
        )

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    def test():
        model.eval()
        output = model(data.x, data.edge_index, edge_weight=data.edge_weight)
        loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
        acc_test = gnn_accuracy(output[data.test_mask], data.y[data.test_mask])
        print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    test()


def train_syn_nc(model, data, device, args):

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

        for epoch in range(args.num_epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, edge_weight=data.edge_weight)
            loss = model.loss(out[data.train_mask], data.y[data.train_mask])
            acc_train = gnn_accuracy(out[data.train_mask], data.y[data.train_mask])
            val_loss = model.loss(out[data.val_mask], data.y[data.val_mask])
            acc_val = gnn_accuracy(out[data.val_mask], data.y[data.val_mask])
            print(
                "Epoch: {:04d}".format(epoch + 1),
                "loss_train: {:.4f}".format(loss.item()),
                "acc_train: {:.4f}".format(acc_train.item()),
                "loss_val: {:.4f}".format(val_loss.item()),
                "acc_val: {:.4f}".format(acc_val.item()),
                "time: {:.4f}s".format(time.time() - t),
            )

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

