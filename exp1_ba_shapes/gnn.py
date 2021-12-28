import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import os

#### GNN Model #####
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, num_layers, hidden_dim):
        super().__init__()
        self.num_node_features, self.num_classes, self.num_layers, self.hidden_dim = num_node_features, num_classes, num_layers, hidden_dim
        self.layers = torch.nn.ModuleList()
        current_dim = self.num_node_features
        for l in range(self.num_layers - 1):
            self.layers.append(GCNConv(current_dim, hidden_dim))
            current_dim = hidden_dim
        self.layers.append(GCNConv(current_dim, self.num_classes))

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


####### GNN Training #######
def train(model, data, device, n_epochs = 200):

    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    # scheduler = ReduceLROnPlateau(optimizer, 'min')
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.99)

    val_err = []
    train_err = []

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])

        if epoch % 10 == 0:
            val_err.append(val_loss.item())
            train_err.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        # scheduler.step(val_loss)

    plt.figure()
    plt.plot(range(n_epochs // 10), val_err)
    plt.plot(range(n_epochs // 10), train_err)

def test(model, data):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
    return(acc)


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