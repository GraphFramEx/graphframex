import os
import json
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

    explain_name = 'pgmexplainer' #'gnnexplainer'

    ### Create data + save data
    n_basis, n_shapes = 2000, 200
    G, labels, plugins = build_function(n_basis, n_shapes)
    data = process_input_data(G, labels)

    data_filename = os.path.join(data_save_dir, data_name) + '.pt'

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
        torch.save(data, data_filename)
    else:
        ### Load data
        data = torch.load(data_filename)

    ### Init GNN model and train on data + save model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features, num_classes, num_layers = data.num_node_features, data.num_classes, num_layers

    model_filename = os.path.join(model_save_dir, data_name) + "/gcn.pth.tar"

    if not os.path.exists(os.path.join(model_save_dir, data_name)):
        os.makedirs(os.path.join(model_save_dir, data_name))
        model = GCN(num_node_features, num_classes, num_layers).to(device)

        train(model, data, device, n_epochs = num_epochs)
        test(model, data)
        torch.save(model.state_dict(), model_filename)

    else:
        ### Load model
        model = GCN(num_node_features, num_classes, num_layers)
        model.load_state_dict(torch.load(model_filename))

    ### Create GNNExplainer
    list_node_idx = range(n_basis, n_basis + 5 * n_shapes)

    F1_scores, GED, Recall, Precision = [], [], [], []

    explain_function = eval('explain_' + explain_name)
    for node_idx in list_node_idx:
        edge_mask = explain_function(model, node_idx, data.x, data.edge_index, labels[node_idx], device)
        recall, precision, f1_score, ged = evaluate(node_idx, data, edge_mask, num_top_edges=6)
        Recall.append(recall)
        Precision.append(precision)
        F1_scores.append(f1_score)
        GED.append(ged)
        print("f1_score, ged", f1_score, ged)

    ### get results + save them
    print(np.mean(F1_scores), np.mean(GED), np.mean(Recall), np.mean(Precision))
    # extract summary at results path and add a line in to the dict
    res_save_dir = 'result/ba_shapes'
    res_filename = os.path.join(res_save_dir, 'summary.json')

    stats = []
    entry = {'explainer':explain_name, 'f1_score': np.mean(F1_scores), 'ged': np.mean(GED), 'recall': np.mean(Recall),
                                        'precision': np.mean(Precision)}

    if not os.path.isfile(res_filename):
        stats.append(entry)
        with open(res_filename, mode='w') as f:
            f.write(json.dumps(stats, indent=2))
    else:
        with open(res_filename) as summary:
            feeds = json.load(summary)

        feeds.append(entry)
        with open(res_filename, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))


if __name__ == '__main__':
    main()