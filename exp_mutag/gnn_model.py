import torch
import torch.nn.functional as F


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            add_self=False,
            normalize_embedding=False,
            dropout=0.0,
            bias=True,
            gpu=True,
            att=False,
    ):
        super(GraphConv, self).__init__()
        self.gpu = gpu
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not gpu:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim)
                )
            if att:
                self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            if add_self:
                self.self_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, output_dim).cuda()
                )
            if att:
                self.att_weight = nn.Parameter(
                    torch.FloatTensor(input_dim, input_dim).cuda()
                )
        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        # deg = torch.sum(adj, -1, keepdim=True)
        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            # import pdb
            # pdb.set_trace()
            att = x_att @ x_att.permute(0, 2, 1)
            # att = self.softmax(att)
            adj = adj * att
            
        if self.gpu:
            adj = adj.cuda()
            x = x.cuda()
            self.weight = nn.Parameter(self.weight.cuda())
            
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        # if self.normalize_embedding:
        # y = F.normalize(y, p=2, dim=2)
        # print(y[0][0])
        return y, adj

class GcnEncoderGraph(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=[],
            concat=True,
            bn=True,
            dropout=0.0,
            add_self=False,
            args=None,
    ):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = add_self
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        self.gpu = args.gpu
        if args.method == "att":
            self.att = True
        else:
            self.att = False
        # if args is not None:
        # self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.att:
                    init.xavier_uniform_(
                        m.att_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.add_self:
                    init.xavier_uniform_(
                        m.self_weight.data, gain=nn.init.calculate_gain("relu")
                    )
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            num_layers,
            add_self,
            normalize=False,
            dropout=0.0,
    ):
        conv_first = GraphConv(
            input_dim=input_dim,
            output_dim=hidden_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        conv_block = nn.ModuleList(
            [
                GraphConv(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    add_self=add_self,
                    normalize_embedding=normalize,
                    dropout=dropout,
                    bias=self.bias,
                    gpu=self.gpu,
                    att=self.att,
                )
                for i in range(num_layers - 2)
            ]
        )
        conv_last = GraphConv(
            input_dim=hidden_dim,
            output_dim=embedding_dim,
            add_self=add_self,
            normalize_embedding=normalize,
            bias=self.bias,
            gpu=self.gpu,
            att=self.att,
        )
        return conv_first, conv_block, conv_last

    def build_pred_layers(
            self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1
    ):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        """ For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        """
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        if self.gpu:
            out = out_tensor.unsqueeze(2).cuda()
        else:
            out = out_tensor.unsqueeze(2)
        return out

    def apply_bn(self, x):
        """ Batch normalization of 3D tensor x
        """
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def gcn_forward(
            self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None
    ):

        """ Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
            The embedding dim is self.pred_input_dim
        """
        x, adj_att = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        # out_all = []
        # out, _ = torch.max(x, dim=1)
        # out_all.append(out)
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)
        x_all.append(x)
        adj_att_all.append(adj_att)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)
        return x_tensor, adj_att_tensor

    def forward_batch(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x, adj_att = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        adj_att_all = [adj_att]
        for i in range(self.num_layers - 2):
            x, adj_att = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
            adj_att_all.append(adj_att)
        x, adj_att = self.conv_last(x, adj)
        adj_att_all.append(adj_att)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        # adj_att_tensor: [batch_size x num_nodes x num_nodes x num_gc_layers]
        adj_att_tensor = torch.stack(adj_att_all, dim=3)

        self.embedding_tensor = output
        ypred = self.pred_model(output)
        # print(output.size())
        return ypred, adj_att_tensor
    
    def forward(self, x, edge_index, batch_num_nodes=None, **kwargs):
        # Encoder Node receives no batch - only one graph
        if x.ndim<3:
            x = x.expand(1,-1,-1)
            edge_index = edge_index.expand(1,-1,-1)
        adj=[]
        for i in range(len(x)):
            max_n = torch.Tensor(x[i]).size(0)
            adj.append(from_edge_index_to_adj(edge_index[i], max_n))
        adj = torch.stack(adj)
        
        if self.gpu:
            adj = adj.cuda()
        pred, adj_att = self.forward_batch(x, adj, batch_num_nodes, **kwargs)
        return pred

    def loss(self, pred, label, type="softmax"):
        # softmax + CE
        if type == "softmax":
            return F.cross_entropy(pred, label, size_average=True)
        elif type == "margin":
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

        # return F.binary_cross_entropy(F.sigmoid(pred[:,0]), label.float())


class GcnEncoderNode(GcnEncoderGraph):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims=[],
            concat=True,
            bn=True,
            dropout=0.0,
            args=None,
    ):
        super(GcnEncoderNode, self).__init__(
            input_dim,
            hidden_dim,
            embedding_dim,
            label_dim,
            num_layers,
            pred_hidden_dims,
            concat,
            bn,
            dropout,
            args=args,
        )
        # if hasattr(args, "loss_weight"):
        # print("Loss weight: ", args.loss_weight)
        # self.celoss = nn.CrossEntropyLoss(weight=args.loss_weight)
        # else:
        self.celoss = nn.CrossEntropyLoss()
        self.gpu = args.gpu

    def forward_batch(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        self.adj_atts = []
        self.embedding_tensor, adj_att = self.gcn_forward(
            x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask
        )
        if self.gpu:
            self.embedding_tensor = self.embedding_tensor.cuda()
            self.pred_model = self.pred_model.cuda()
        pred = self.pred_model(self.embedding_tensor)
        return pred, adj_att

    def forward(self, x, edge_index, batch_num_nodes=None, **kwargs):
        # Encoder Node receives no batch - only one graph
        max_n = x.size(0)
        adj = from_edge_index_to_adj(edge_index, max_n)
        if self.gpu:
            adj = adj.cuda()
        pred, adj_att = self.forward_batch(x.expand(1, -1, -1), adj.expand(1, -1, -1), batch_num_nodes, **kwargs)
        ypred = torch.squeeze(pred, 0)
        return ypred

    def loss(self, pred, label):
        # Transpose if batch dim:
        # pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)

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

