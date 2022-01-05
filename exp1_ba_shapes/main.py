import sys, os
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
import pickle
import time
from datetime import datetime
import argparse
import random

from dataset import *
from evaluate import *
from explainer import *
from gnn import GCN, train, test
from utils import check_dir, normalize_masks
from config.params import dictmerge



def main(args):

    np.random.seed(args.seed)
    random.seed(args.seed)
    build_function = eval('build_' + args.data_name)
    is_true_label = eval(args.true_label)

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
        model.eval()
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

    ### Store results in summary.json

    date = datetime.now().strftime("%Y_%m_%d")
    res_save_dir = os.path.join('result', args.data_name)
    res_filename = os.path.join(res_save_dir, f'summary_{date}.json')
    check_dir(res_save_dir)


    # explain only nodes for each the GCN made accurate predictions
    pred_labels = model(data.x, data.edge_index).argmax(dim=1)
    list_node_idx = np.where(pred_labels == data.y)[0]
    list_node_idx_house = list_node_idx[list_node_idx > args.n_basis]
    list_test_nodes = [x.item() for x in random.choices(list_node_idx_house, k=args.num_test_nodes)]
    targets = pred_labels # here using true_labels or pred_labels is equivalent for nodes in list_test_nodes

    def compute_edge_masks(explainer_name, list_test_nodes, model, data, targets, device):
        explain_function = eval('explain_' + explainer_name)
        Time = []
        edge_masks = []
        for node_idx in list_test_nodes:
            x = torch.FloatTensor(data.x.detach().numpy().copy())
            edge_index = torch.LongTensor(data.edge_index.detach().numpy().copy())
            start_time = time.time()
            edge_mask = explain_function(model, node_idx, x, edge_index, targets[node_idx], device)
            end_time = time.time()
            duration_seconds = end_time - start_time
            Time.append(duration_seconds)
            edge_masks.append(edge_mask)
        return edge_masks, Time

    if args.explainer_name == 'subgraphx':
        hard = True
    else:
        hard = False


    #### Save edge_masks
    # save
    mask_save_dir = os.path.join(res_save_dir, f'masks_{date}')
    check_dir(mask_save_dir)

    mask_filename = os.path.join(mask_save_dir, f'masks_{args.explainer_name}.pickle')

    if os.path.isfile(mask_filename):
        # open
        with open(mask_filename, 'rb') as handle:
            edge_masks, Time = tuple(pickle.load(handle))
    else:
        edge_masks, Time = compute_edge_masks(args.explainer_name, list_test_nodes, model, data, targets, device)
        with open(mask_filename, 'wb') as handle:
            pickle.dump((edge_masks, Time), handle)

    print('edge_masks', edge_masks)
    print('Time', Time)
    edge_masks_norm = normalize_masks(edge_masks)

    accuracy = eval_accuracy(data, edge_masks, list_test_nodes, num_top_edges=args.num_top_edges, is_hard_mask=hard)
    soft_related_preds = eval_related_pred(model, data, edge_masks_norm, list_test_nodes, hard_mask=False)
    hard_related_preds = eval_related_pred(model, data, edge_masks_norm, list_test_nodes, hard_mask=True, num_top_edges=args.num_top_edges)
    fidelity_soft = {k+'_soft':v for k,v in eval_fidelity(soft_related_preds).items()}
    fidelity_hard = {k+'_hard':v for k,v in eval_fidelity(hard_related_preds).items()}
    infos = {"explainer": args.explainer_name, "num_test_nodes": args.num_test_nodes, "groundtruth target": is_true_label, "time": float(format(np.mean(Time), '.4f'))}

    ### get results + save them
    print("__infos:" + json.dumps(infos))
    print("__accuracy:" + json.dumps(accuracy))
    print("__fidelity_soft:" + json.dumps(fidelity_soft))
    print("__fidelity_hard:" + json.dumps(fidelity_hard))


    # extract summary at results path and add a line in to the dict
    stats = []
    entry = dict(list(infos.items()) + list(accuracy.items()) + list(fidelity_soft.items()) + list(fidelity_hard.items()))
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

    parser.add_argument('--seed', help='random seed', type=int, default=41)
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
    parser.add_argument('--num_test_nodes', help='number of testing nodes', type=int, default=3)
    parser.add_argument('--num_top_edges', help='number of edges to keep in explanation', type=int, default=6)
    parser.add_argument('--true_label', help='do you take target as true label or predicted label', type=str,
                        default='True')
    parser.add_argument('--explainer_name', help='explainer', type=str, default='gnnexplainer')

    args = parser.parse_args()

    main(args)