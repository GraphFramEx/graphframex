import sys, os
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import time
from datetime import datetime

from dataset import *
from evaluate import evaluate
from explainer import *
from gnn import GCN, train, test
from utils import check_dir


def main(args):

    build_function = eval('build_' + args.data_name)

    EXPLAIN_LIST = [
        'subgraphx']  # ['random', 'distance', 'pagerank', 'gradcam', 'sa_node', 'ig_node', 'occlusion', 'gnnexplainer', 'pgmexplainer', 'subgraphx']

    check_dir(args.data_save_dir)
    data_filename = os.path.join(args.data_save_dir, args.data_name) + f'{args.data_name}.pt'

    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
        true_labels = data.y
    else:
        G, true_labels, plugins = build_function(args.n_basis, args.n_shapes)
        data = process_input_data(G, true_labels)
        torch.save(data, data_filename)

    ### Init GNN model and train on data + save model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features, num_classes = data.num_node_features, data.num_classes

    check_dir(os.path.join(args.model_save_dir, args.data_name))
    model_filename = os.path.join(args.model_save_dir, args.data_name) + f"/gcn_{args.num_layers}.pth.tar"

    if os.path.isfile(model_filename):
        model = GCN(num_node_features, num_classes, args.num_layers, args.hidden_dim)
        ckpt = torch.load(model_filename)
        model.load_state_dict(ckpt['model_state'])
    else:
        model = GCN(num_node_features, num_classes, args.num_layers, args.hidden_dim).to(device)
        train(model, data, device, n_epochs=args.num_epochs)
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
    list_node_idx = range(n_basis, n_basis + 10)  # * n_shapes)

    ### Store results in summary.json

    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    res_save_dir = os.path.join('result', args.data_name)
    res_filename = os.path.join(res_save_dir, f'summary_{date}.json')

    check_dir(res_save_dir)

    stats = []

    Time = []
    F1_scores, GED, Recall, Precision, AUC = [], [], [], [], []

    explain_function = eval('explain_' + args.explainer_name)
    for node_idx in list_node_idx:

        if args.true_label:
            target = true_labels[node_idx]
        else:
            target = torch.argmax(model(data.x, data.edge_index)[node_idx])

        start_time = time.time()
        edge_mask = explain_function(model, node_idx, data.x, data.edge_index, target, device)
        end_time = time.time()
        duration_seconds = end_time - start_time
        Time.append(duration_seconds)

        if args.explainer_name == 'subgraphx':
            hard = True
        else:
            hard = False
        recall, precision, f1_score, ged, auc = evaluate(node_idx, data, edge_mask, num_top_edges=6,
                                                         is_hard_mask=hard)

        Recall.append(recall)
        Precision.append(precision)
        F1_scores.append(f1_score)
        GED.append(ged)
        AUC.append(auc)
        print("f1_score, ged, auc", f1_score, ged, auc)

    ### get results + save them
    print(np.mean(F1_scores), np.mean(GED), np.mean(Recall), np.mean(Precision), np.mean(AUC))
    print("__score:" + json.dumps({
        "explainer": args.explainer_name, "groundtruth target": args.true_label,
             "auc": float(format(np.mean(AUC), '.4f')), "f1_score": float(format(np.mean(F1_scores), '.4f')),
             "ged": float(format(np.mean(GED), '.2f')), "recall": float(format(np.mean(Recall), '.2f')),
             "precision": float(format(np.mean(Precision), '.2f')), "time": float(format(np.mean(Time), '.2f'))
    }))
    # extract summary at results path and add a line in to the dict
    entry = {'explainer': args.explainer_name, 'groundtruth target': args.true_label,
             'auc': float(format(np.mean(AUC), '.4f')), 'f1_score': float(format(np.mean(F1_scores), '.4f')),
             'ged': float(format(np.mean(GED), '.2f')), 'recall': float(format(np.mean(Recall), '.2f')),
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

