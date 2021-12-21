import os
import json
import time
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
    hidden_dim = 16
    num_epochs = 200
    model_save_dir = 'model'

    EXPLAIN_LIST = ['random', 'distance', 'pagerank', 'gradcam', 'sa_node', 'ig_node', 'occlusion', 'gnnexplainer', 'pgmexplainer', 'subgraphx']

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
    num_node_features, num_classes = data.num_node_features, data.num_classes

    model_filename = os.path.join(model_save_dir, data_name) + "/gcn.pth.tar"

    if not os.path.exists(os.path.join(model_save_dir, data_name)):
        os.makedirs(os.path.join(model_save_dir, data_name))
        model = GCN(num_node_features, num_classes, num_layers, hidden_dim).to(device)

        train(model, data, device, n_epochs=num_epochs)
        test(model, data)
        torch.save(model.state_dict(), model_filename)

    else:
        ### Load model
        model = GCN(num_node_features, num_classes, num_layers, hidden_dim)
        model.load_state_dict(torch.load(model_filename))

    ### Create GNNExplainer
    list_node_idx = range(n_basis, n_basis + 50) #* n_shapes)

    ### Store results in summary.json
    res_save_dir = os.path.join('result', data_name)
    res_filename = os.path.join(res_save_dir, 'summary.json')

    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)

    stats = []

    for explain_name in EXPLAIN_LIST:
        Time = []
        F1_scores, GED, Recall, Precision = [], [], [], []

        explain_function = eval('explain_' + explain_name)
        for node_idx in list_node_idx:
            start_time = time.time()
            edge_mask = explain_function(model, node_idx, data.x, data.edge_index, labels[node_idx], device)
            end_time = time.time()
            duration_seconds = end_time - start_time
            Time.append(duration_seconds)

            if explain_name == 'subgraphx':
                hard = True
            else:
                hard = False
            recall, precision, f1_score, ged = evaluate(node_idx, data, edge_mask, num_top_edges=6, is_hard_mask=hard)

            Recall.append(recall)
            Precision.append(precision)
            F1_scores.append(f1_score)
            GED.append(ged)
            print("f1_score, ged", f1_score, ged)

        ### get results + save them
        print(np.mean(F1_scores), np.mean(GED), np.mean(Recall), np.mean(Precision))
        # extract summary at results path and add a line in to the dict
        entry = {'explainer':explain_name, 'f1_score': float(format(np.mean(F1_scores), '.4f')), 'ged': float(format(np.mean(GED), '.2f')), 'recall': float(format(np.mean(Recall), '.2f')),
                                            'precision': float(format(np.mean(Precision), '.2f')), 'time': float(format(np.mean(Time), '.2f'))}

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