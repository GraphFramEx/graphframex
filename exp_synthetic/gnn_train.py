import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
import sklearn.metrics as metrics
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from gen_utils import from_edge_index_to_adj, from_adj_to_edge_index


####### GNN Training #######
def train(model, data, device, args):

    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.96)

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

        loss.backward()
        optimizer.step()
        # scheduler.step()
        # scheduler.step(val_loss)

    #plt.figure()
    #plt.plot(range(args.num_epochs // 10), val_err)
    #plt.plot(range(args.num_epochs // 10), train_err)




def save_model(model, args):
    filename = os.path.join(args.save_dir, args.dataset) + "gcn.pth.tar"
    torch.save(
        {
            "model_type": 'gcn',
            "model_state": model.state_dict()
        },
        str(filename),
    )

def load_model(args):
    '''Load a pre-trained pytorch model from checkpoint.
        '''
    print("loading model")
    filename = os.path.join(args.save_dir, args.dataset) + "/gcn.pth.tar"
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train_gnn.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt

