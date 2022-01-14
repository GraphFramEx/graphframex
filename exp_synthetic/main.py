import sys, os
sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import json
import pickle
import time
from datetime import datetime
import argparse
import random
import itertools

from dataset import *
from evaluate import *
from explainer import *
from gnn import GcnEncoderNode, train, gnn_scores, get_labels
from gen_utils import check_dir, get_subgraph
from parser_utils import arg_parse


def main(args):

    random.seed(args.seed)
    is_true_label = eval(args.true_label)

    check_dir(args.data_save_dir)
    subdir = os.path.join(args.data_save_dir, args.dataset)
    check_dir(subdir)
    data_filename = os.path.join(subdir, f'{args.dataset}.pt')

    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
    else:
        data = build_data(args)
        torch.save(data, data_filename)

    ### Init GNN model and train on data + save model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    check_dir(os.path.join(args.model_save_dir, args.dataset))
    model_filename = os.path.join(args.model_save_dir, args.dataset) + f"/gcn_{args.num_gc_layers}.pth.tar"

    def get_data_args(data, args):
        args.num_classes = data.num_classes
        args.input_dim = data.x.size(1)
        if args.dataset == 'syn1':
            args.num_top_edges = 6
        elif args.dataset == 'syn2':
            args.num_top_edges = 6
        elif args.dataset == 'syn3':
            args.num_top_edges = 12
        elif args.dataset == 'syn4':
            args.num_top_edges = 6
        elif args.dataset == 'syn5':
            args.num_top_edges = 12
        elif args.dataset == 'syn6':
            args.num_top_edges = 5
        return args


    args = get_data_args(data, args)
    print('args.num_classes', args.num_classes)

    if os.path.isfile(model_filename):
        model = GcnEncoderNode(args.input_dim,
                               args.hidden_dim,
                               args.output_dim,
                               args.num_classes,
                               args.num_gc_layers, args=args)

    else:
        model = GcnEncoderNode(args.input_dim,
                               args.hidden_dim,
                               args.output_dim,
                               args.num_classes,
                               args.num_gc_layers, args=args)
        train(model, data, device, args)
        model.eval()
        results_train, results_test = gnn_scores(model, data)
        torch.save(
            {
                "model_type": 'gcn',
                "results_train": results_train,
                "results_test": results_test,
                "model_state": model.state_dict()
            },
            str(model_filename),
        )
        #torch.save(model.state_dict(), model_filename)

    ckpt = torch.load(model_filename)
    model.load_state_dict(ckpt['model_state'])
    print("__gnn_train_scores: ", ckpt['results_train'])
    print("__gnn_test_scores: ", ckpt['results_test'])
    #print("__gnn_train_scores:" + json.dumps(ckpt['results_train']))
    #print("__gnn_test_scores:" + json.dumps(ckpt['results_test']))
    model.eval()


    ### Store results in summary.json

    date = datetime.now().strftime("%Y_%m_%d")
    res_save_dir = os.path.join('result', args.dataset)
    res_filename = os.path.join(res_save_dir, f'summary_{date}.json')
    check_dir(res_save_dir)


    # explain only nodes for each the GCN made accurate predictions
    pred_labels = get_labels(model(data.x, data.edge_index))
    list_node_idx = np.where(pred_labels == data.y)[0]
    list_node_idx_house = list_node_idx[list_node_idx > args.num_basis]
    #list_test_nodes = list_node_idx_house[:args.num_test_nodes].tolist()
    list_test_nodes = [x.item() for x in random.choices(list_node_idx_house, k=args.num_test_nodes)]
    targets = data.y # here using true_labels or pred_labels is equivalent for nodes in list_test_nodes

    #list_test_nodes = range(args.num_basis,args.num_basis+args.num_test_nodes)
    print('length test nodes', len(list_test_nodes))
    def compute_edge_masks(explainer_name, list_test_nodes, model, data, targets, device):
        explain_function = eval('explain_' + explainer_name)
        Time = []
        edge_masks = []
        for node_idx in list_test_nodes:
            x = torch.FloatTensor(data.x.detach().numpy().copy())
            edge_index = torch.LongTensor(data.edge_index.detach().numpy().copy())
            start_time = time.time()
            edge_mask = explain_function(model, node_idx, x, edge_index, targets[node_idx], device, args)
            end_time = time.time()
            duration_seconds = end_time - start_time
            Time.append(duration_seconds)
            edge_masks.append(edge_mask)
        return edge_masks, Time


    #### Save edge_masks
    # save
    mask_save_dir = os.path.join(res_save_dir, f'masks_{args.dataset}_{date}')
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

    infos = {"dataset": args.dataset, "explainer": args.explainer_name, "num_test_nodes": args.num_test_nodes,
             "groundtruth target": is_true_label, "time": float(format(np.mean(Time), '.4f'))}
    print("__infos:" + json.dumps(infos))
    accuracy = eval_accuracy(data, edge_masks, list_test_nodes, args)
    print("__accuracy:" + json.dumps(accuracy))

    FIDELITY_SCORES = {}
    FIDELITY_SUB_SCORES = {}
    list_params = {'sparsity':[0.7], 'normalize': [True], 'hard_mask': [True]}
    keys, values = zip(*list_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for i, params in enumerate(permutations_dicts):
        related_preds = eval_related_pred(model, data, edge_masks, list_test_nodes, params)
        related_preds_sub = eval_related_pred_subgraph(model, data, edge_masks, list_test_nodes, params)

        labels = related_preds['true_label']
        ori_probs = np.choose(labels, related_preds['origin'].T)
        important_probs = np.choose(labels, related_preds['masked'].T)
        unimportant_probs = np.choose(labels, related_preds['maskout'].T)
        probs_summary = {'ori_probs': ori_probs.mean().item(), 'unimportant_probs': unimportant_probs.mean().item(), 'important_probs': important_probs.mean().item()}
        print("__pred:" + json.dumps(probs_summary))

        fidelity = eval_fidelity(related_preds, params)
        print("__fidelity:" + json.dumps(fidelity))

        fidelity_sub = eval_fidelity(related_preds_sub, params)
        print("__fidelity_sub:" + json.dumps(fidelity_sub))

        FIDELITY_SCORES[i] = fidelity
        FIDELITY_SUB_SCORES[i] = fidelity_sub


    # extract summary at results path and add a line in to the dict
    stats = []
    entry = dict(list(infos.items()) + list(accuracy.items()) + list(FIDELITY_SCORES.items()) + list(FIDELITY_SUB_SCORES.items()))
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
    args = arg_parse()
    main(args)