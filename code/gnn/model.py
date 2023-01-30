import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, GINEConv, TransformerConv
from torch_geometric.data.batch import Batch
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool


def get_gnnNets(input_dim, output_dim, model_params):
    if model_params["model_name"].lower() in [
        "base",
        "gcn",
        "gat",
        "gin",
        "transformer",
    ]:
        GNNmodel = model_params["model_name"].upper()
        return eval(GNNmodel)(
            input_dim=input_dim, output_dim=output_dim, model_params=model_params
        )
    else:
        raise ValueError(
            f"GNN name should be gcn " f"and {model_params.gnn_name} is not defined."
        )


def identity(x: torch.Tensor, batch: torch.Tensor):
    return x


def cat_max_sum(x, batch):
    node_dim = x.shape[-1]
    num_node = 25
    x = x.reshape(-1, num_node, node_dim)
    return torch.cat([x.max(dim=1)[0], x.sum(dim=1)], dim=-1)


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool,
        "identity": identity,
        "cat_max_sum": cat_max_sum,
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    return readout_func_dict[readout.lower()]


class GNNPool(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = get_readout_layers(readout)

    def forward(self, x, batch):
        return self.readout(x, batch)


class GNNBase(nn.Module):
    def __init__(self):
        super(GNNBase, self).__init__()

    def _argsparse(self, *args, **kwargs):
        r"""Parse the possible input types.
        If the x and edge_index are in args, follow the args.
        In other case, find them in kwargs.
        """
        if args:
            if len(args) == 1:
                data = args[0]
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                else:
                    edge_attr = torch.ones(
                        edge_index.shape[1], dtype=torch.float32, device=x.device
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 2:
                x, edge_index = args[0], args[1]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_attr = torch.ones(
                    edge_index.shape[1], dtype=torch.float32, device=x.device
                )

            elif len(args) == 3:
                x, edge_index, edge_attr = args[0], args[1], args[2]
                batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)

            elif len(args) == 4:
                x, edge_index, edge_attr, batch = args[0], args[1], args[2], args[3]
            else:
                raise ValueError(
                    f"forward's args should take 1, 2 or 3 arguments but got {len(args)}"
                )
        else:
            data: Batch = kwargs.get("data")
            if not data:
                x = kwargs.get("x")
                edge_index = kwargs.get("edge_index")
                assert (
                    x is not None
                ), "forward's args is empty and required node features x is not in kwargs"
                assert (
                    edge_index is not None
                ), "forward's args is empty and required edge_index is not in kwargs"
                batch = kwargs.get("batch")
                if not batch:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
                edge_attr = kwargs.get("edge_attr")
                if not edge_attr:
                    edge_attr = torch.ones(
                        edge_index.shape[1], dtype=torch.float32, device=x.device
                    )

            else:
                x = data.x
                edge_index = data.edge_index
                if hasattr(data, "edge_attr"):
                    edge_attr = data.edge_attr
                    if edge_attr is None:
                        edge_attr = torch.ones(
                            edge_index.shape[1], dtype=torch.float32, device=x.device
                        )
                else:
                    edge_attr = torch.ones(
                        edge_index.shape[1], dtype=torch.float32, device=x.device
                    )
                if hasattr(data, "batch"):
                    batch = data.batch
                    if batch is None:
                        batch = torch.zeros(
                            x.shape[0], dtype=torch.int64, device=x.device
                        )
                else:
                    batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        return x, edge_index, edge_attr, batch


class GNN_basic(GNNBase):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
    ):
        super(GNN_basic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = model_params["num_layers"]
        self.hidden_dim = model_params["hidden_dim"]
        self.dropout = model_params["dropout"]
        # readout
        self.readout = model_params["readout"]
        self.readout_layer = GNNPool(self.readout)
        self.get_layers()

    def get_layers(self):
        # GNN layers
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(NNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return

    def forward(self, *args, **kwargs):
        _, _, _, batch = self._argsparse(*args, **kwargs)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = self.mlps(x)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)

    def get_emb(self, *args, **kwargs):
        x, edge_index, edge_attr, _ = self._argsparse(*args, **kwargs)
        for layer in self.convs:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return x

    def get_pred_label(self, pred):
        return pred.argmax(dim=1)


class GAT(GNN_basic):
    def __init__(self, input_dim, output_dim, model_params):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )
        self.edge_dim = model_params["edge_dim"]

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GATConv(current_dim, self.hidden_dim, edge_dim=self.edge_dim)
            )
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return


class GCN(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(GCNConv(current_dim, self.hidden_dim))
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return


class GIN(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(current_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    ),
                    edge_dim=self.edge_dim,
                )
            )
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return


class Transformer(GNN_basic):
    def __init__(
        self,
        input_dim,
        output_dim,
        model_params,
    ):
        super().__init__(
            input_dim,
            output_dim,
            model_params,
        )

    def get_layers(self):
        self.convs = nn.ModuleList()
        current_dim = self.input_dim
        for l in range(self.num_layers):
            self.convs.append(
                TransformerConv(current_dim, self.hidden_dim, edge_dim=self.edge_dim)
            )
            current_dim = self.hidden_dim
        # FC layers
        self.mlps = nn.Linear(current_dim, self.output_dim)
        return
