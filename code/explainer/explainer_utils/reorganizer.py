import torch
import copy
from tqdm import tqdm
from torch_geometric.data import DataLoader


def filter_correct_data(model, dataset, loader, flag='Training', batch_size=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_mask = torch.zeros(len(loader.dataset), dtype=torch.bool)
    idx = 0
    for g in tqdm(iter(loader), total=len(loader)):
        g.to(device)
        if g.y == model(g).argmax(dim=1):
            graph_mask[idx] = True
        idx += 1

    loader = DataLoader(dataset[graph_mask], batch_size=1, shuffle=False)
    print("number of graphs in the %s:%4d" % (flag, graph_mask.nonzero().size(0)))
    return dataset, loader

def filter_correct_data_batch(model, dataset, loader, flag='training', batch_size=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    graph_mask = []
    for g in tqdm(iter(loader), total=len(loader)):
        g.to(device)
        tmp = (g.y == model(g).argmax(dim=1))
        graph_mask += tmp.tolist()

    # must convert the graph_mask to the boolean tensor.
    graph_mask = torch.BoolTensor(graph_mask)

    shuffle_flag = False
    if flag is 'training':
        shuffle_flag = True

    loader = DataLoader(dataset[graph_mask], batch_size=batch_size, shuffle=shuffle_flag)
    print("number of graphs in the %s:%4d" % (flag, sum(graph_mask)))
    return dataset, loader




def relabel_graph(graph, selection):
    subgraph = copy.deepcopy(graph)

    # retrieval properties of the explanatory subgraph
    # .... the edge_index.
    subgraph.edge_index = graph.edge_index.T[selection].T
    # .... the edge_attr.
    subgraph.edge_attr = graph.edge_attr[selection]
    # .... the nodes.
    sub_nodes = torch.unique(subgraph.edge_index)
    # .... the node features.
    subgraph.x = graph.x[sub_nodes]
    subgraph.batch = graph.batch[sub_nodes]

    row, col = graph.edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((graph.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    subgraph.edge_index = node_idx[subgraph.edge_index]

    return subgraph