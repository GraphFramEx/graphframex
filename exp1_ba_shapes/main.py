import sys, os
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import time
from datetime import datetime
import argparse

from dataset import *
from evaluate import evaluate
from explainer import *
from gnn import GCN, train, test
from utils import check_dir


def main(args):

    build_function = eval('build_' + args.data_name)
    is_true_label = eval(args.true_label)

    EXPLAIN_LIST = [
        'subgraphx']  # ['random', 'distance', 'pagerank', 'gradcam', 'sa_node', 'ig_node', 'occlusion', 'gnnexplainer', 'pgmexplainer', 'subgraphx']

    check_dir(args.data_save_dir)
    subdir = os.path.join(args.data_save_dir, args.data_name)
    check_dir(subdir)
    data_filename = os.path.join(subdir, f'{args.data_name}.pt')

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
    list_node_idx = np.random.randint(args.n_basis, args.n_basis + 5*args.n_shapes, args.num_test_nodes)

    ### Store results in summary.json

    date = datetime.now().strftime("%Y_%m_%d")
    res_save_dir = os.path.join('result', args.data_name)
    res_filename = os.path.join(res_save_dir, f'summary_{date}.json')

    check_dir(res_save_dir)

    stats = []

    Time = []
    F1_scores, GED, Recall, Precision, AUC = [], [], [], [], []

    explain_function = eval('explain_' + args.explainer_name)
    for node_idx in list_node_idx:

        if is_true_label:
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
        print(f"f1_score={f1_score}, ged={ged}, auc={auc}")

    ### get results + save them
    print("__scores:" + json.dumps({
        "explainer": args.explainer_name, "groundtruth target": is_true_label,
             "auc": float(format(np.mean(AUC), '.4f')), "f1_score": float(format(np.mean(F1_scores), '.4f')),
             "ged": float(format(np.mean(GED), '.2f')), "recall": float(format(np.mean(Recall), '.2f')),
             "precision": float(format(np.mean(Precision), '.2f')), "time": float(format(np.mean(Time), '.4f'))
    }))
    # extract summary at results path and add a line in to the dict
    entry = {'explainer': args.explainer_name, 'groundtruth target': is_true_label,
             'auc': float(format(np.mean(AUC), '.4f')), 'f1_score': float(format(np.mean(F1_scores), '.4f')),
             'ged': float(format(np.mean(GED), '.2f')), 'recall': float(format(np.mean(Recall), '.2f')),
             'precision': float(format(np.mean(Precision), '.2f')), 'time': float(format(np.mean(Time), '.4f'))}

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--dest', type=str, default='/Users/kenzaamara/PycharmProjects/Explain')
    # saving data
    parser.add_argument('--data_save_dir', help='File list by write RTL command', type=str, default='data')
    parser.add_argument('--data_name', type=str, default='ba_shapes')

    # build ba-shape graphs
    parser.add_argument('--n_basis', help='number of nodes in graph', type=int, default=2000)
    parser.add_argument('--n_shapes', help='number of houses', type=int, default=200)

    # gnn achitecture parameters
    parser.add_argument('--num_layers', help='number of GCN layers', type=int, default=2)
    parser.add_argument('--hidden_dim', help='number of neurons in hidden layers', type=int, default=16)

    # training parameters
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=200)

    # saving model
    parser.add_argument('--model_save_dir', help='saving directory for gnn model', type=str, default='model')

    # explainer params
    parser.add_argument('--num_test_nodes', help='number of testing nodes', type=int, default=200)
    parser.add_argument('--true_label', help='do you take target as true label or predicted label', type=str,
                        default='True')
    parser.add_argument('--explainer_name', help='explainer', type=str, default='random')

    args = parser.parse_args()

    main(args)