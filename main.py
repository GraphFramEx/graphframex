import os
import numpy as np
import torch
from dataset import *
from gnn import GCN, train, test
from explainer import *
from evaluate import evaluate

def main():

    # args: data_save_dir, data_name
    # args: n_basis, n_shapes
    # args_model: num_layers
    # args_training: num_epochs,

    data_save_dir = 'data'
    data_name = 'ba_shapes'
    build_function = eval('build_' + data_name)
    num_layers = 2
    num_epochs = 200
    model_save_dir = 'model'

    explain_name = 'gnnexplainer'

    ### Create data + save data
    n_basis, n_shapes = 2000, 200
    G, labels, plugins = build_function(n_basis, n_shapes)
    data = process_input_data(G, labels)
    data_filename = os.path.join(data_save_dir, data_name) + '.pt'
    torch.save(data, data_filename)

    ### Load data
    data = torch.load(data_filename)

    ### Init GNN model and train on data + save model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features, num_classes, num_layers = data.num_node_features, data.num_classes, num_layers
    model = GCN(num_node_features, num_classes, num_layers).to(device)

    train(model, data, device, n_epochs = num_epochs)
    test(model, data)

    filename = os.path.join(model_save_dir, data_name) + "/gcn.pth.tar"
    torch.save(model.state_dict(), filename)


    ### Load model
    model = GCN(num_node_features, num_classes, num_layers)
    model.load_state_dict(torch.load(filename))
    ### Create GNNExplainer
    list_node_idx = range(n_basis, n_basis + 5 * n_shapes)

    F1_scores, GED, Recall, Precision = [], [], [], []

    explain_function = eval('explain_' + explain_name)
    for node_idx in list_node_idx:
        edge_mask = explain_function(model, node_idx, data.x, data.edge_index, labels[node_idx])
        recall, precision, f1_score, ged = evaluate(node_idx, data, edge_mask, num_top_edges=6)
        Recall.append(recall)
        Precision.append(precision)
        F1_scores.append(f1_score)
        GED.append(ged)
        print("f1_score, ged", f1_score, ged)

    ### get results + save them
    print(np.mean(F1_scores), np.mean(GED), np.mean(Recall), np.mean(Precision))

if __name__ == '__main__':
    main()