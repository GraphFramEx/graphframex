import os
import json
import time
from datetime import datetime
from exp1_ba_shapes.dataset import *
from exp1_ba_shapes.gnn import GCN, train, test
from exp1_ba_shapes.explainer import *
from exp1_ba_shapes.evaluate import evaluate
from utils import check_dir

def main():

    # args: data_save_dir, data_name
    # args: n_basis, n_shapes
    # args_model: num_layers
    # args_training: num_epochs,

    data_save_dir = 'data'
    data_name = 'ba_shapes'
    build_function = eval('build_' + data_name)
    num_layers = 3
    hidden_dim = 16
    num_epochs = 200
    model_save_dir = 'model'

    EXPLAIN_LIST = ['subgraphx']#['random', 'distance', 'pagerank', 'gradcam', 'sa_node', 'ig_node', 'occlusion', 'gnnexplainer', 'pgmexplainer', 'subgraphx']

    ### Create data + save data
    n_basis, n_shapes = 2000, 200

    check_dir(data_save_dir)
    data_filename = os.path.join(data_save_dir, data_name) + f'{data_name}.pt'

    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
        true_labels = data.y
    else:
        G, true_labels, plugins = build_function(n_basis, n_shapes)
        data = process_input_data(G, true_labels)
        torch.save(data, data_filename)


    ### Init GNN model and train on data + save model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features, num_classes = data.num_node_features, data.num_classes

    check_dir(os.path.join(model_save_dir, data_name))
    model_filename = os.path.join(model_save_dir, data_name) + f"/gcn_{num_layers}.pth.tar"

    if os.path.isfile(model_filename):
        model = GCN(num_node_features, num_classes, num_layers, hidden_dim)
        ckpt = torch.load(model_filename)
        model.load_state_dict(ckpt['model_state'])
    else:
        model = GCN(num_node_features, num_classes, num_layers, hidden_dim).to(device)
        train(model, data, device, n_epochs=num_epochs)
        acc = test(model, data)
        torch.save(
            {
                "model_type": 'gcn',
                "acc": float(format(acc, '.4f')),
                "model_state": model.state_dict()
            },
            str(model_filename),
        )
        #torch.save(model.state_dict(), model_filename)


    ### Create GNNExplainer
    list_node_idx = range(n_basis, n_basis + 50) #* n_shapes)

    ### Store results in summary.json

    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    res_save_dir = os.path.join('result', data_name)
    res_filename = os.path.join(res_save_dir, f'summary_{date}.json')

    check_dir(res_save_dir)

    stats = []

    for groundtruth_target in [False]:
        for explain_name in EXPLAIN_LIST:
            Time = []
            F1_scores, GED, Recall, Precision, AUC = [], [], [], [], []

            explain_function = eval('explain_' + explain_name)
            for node_idx in list_node_idx:

                if groundtruth_target:
                    target = true_labels[node_idx]
                else:
                    target = torch.argmax(model(data.x, data.edge_index)[node_idx])

                start_time = time.time()
                edge_mask = explain_function(model, node_idx, data.x, data.edge_index, target, device)
                end_time = time.time()
                duration_seconds = end_time - start_time
                Time.append(duration_seconds)

                if explain_name == 'subgraphx':
                    hard = True
                else:
                    hard = False
                recall, precision, f1_score, ged, auc = evaluate(node_idx, data, edge_mask, num_top_edges=6, is_hard_mask=hard)

                Recall.append(recall)
                Precision.append(precision)
                F1_scores.append(f1_score)
                GED.append(ged)
                AUC.append(auc)
                print("f1_score, ged, auc", f1_score, ged, auc)

            ### get results + save them
            print(np.mean(F1_scores), np.mean(GED), np.mean(Recall), np.mean(Precision), np.mean(AUC))
            # extract summary at results path and add a line in to the dict
            entry = {'explainer':explain_name, 'groundtruth target': groundtruth_target, 'auc':float(format(np.mean(AUC), '.4f')), 'f1_score': float(format(np.mean(F1_scores), '.4f')), 'ged': float(format(np.mean(GED), '.2f')), 'recall': float(format(np.mean(Recall), '.2f')),
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