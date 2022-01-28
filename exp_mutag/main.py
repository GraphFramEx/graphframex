import os
import re
import sys,os
sys.path.append(os.getcwd())

import torch
from torch_geometric.data import download_url


from gen_utils import check_dir
from dataset import extract_zip, extract_gz, process_mutag, collate_data
import math_utils


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
from gnn_model import *
from gnn_train import *
from gnn_eval import *
from data_sampler import *

import random

from gen_utils import check_dir, get_subgraph
from parser_utils import arg_parse



def compute_edge_masks(explainer_name, dataset, model, device):
    explain_function = eval('explain_' + explainer_name)
    edge_index_set = get_edge_index_set(dataset)
    edge_masks_set = []
    Time = []

    for batch_idx, data in enumerate(dataset):
        edge_masks = []
        if args.gpu:
            adj = Variable(data["adj"].float(), requires_grad=False).cuda()
            h0 = Variable(data["feats"].float()).cuda()
            targets = data["label"].long().numpy()
        else:
            adj = Variable(data["adj"].float(), requires_grad=False)
            h0 = Variable(data["feats"].float())
            targets = data["label"].long().numpy()

        for i in range(len(edge_index_set[batch_idx])): 
            start_time = time.time()
            edge_mask = explain_function(model, -1, h0[i], edge_index_set[batch_idx][i], targets[i], device, args)
            end_time = time.time()
            duration_seconds = end_time - start_time
            edge_masks.append(edge_mask)
            Time.append(duration_seconds)
            
        edge_masks_set.append(edge_masks)
    return(edge_masks_set, Time)


def main(args):

    ### Init GNN model and train on data + save model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.gpu = True if torch.cuda.is_available() else False
    
    check_dir(os.path.join(args.model_save_dir, args.dataset))
    model_filename = os.path.join(args.model_save_dir, args.dataset) + f"/gcn_{args.num_gc_layers}.pth.tar"
    
    data_save_dir = os.path.join('data', args.dataset)

    check_dir(data_save_dir)
    raw_data_dir = os.path.join(data_save_dir, 'raw_data')
    # Save data_list
    data_filename = os.path.join(data_save_dir, args.dataset) + '.pt'
    print(model_filename)
    print(data_filename)
    #download MUTAG from url and put it in raw_dir
    url = 'https://github.com/divelab/DIG_storage/raw/main/xgraph/datasets/MUTAG.zip'

    path = download_url(url, raw_data_dir)
    if url[-2:] == 'gz':
        extract_gz(path, raw_data_dir)
        os.unlink(path)
    elif url[-3:] == 'zip':
        extract_zip(path, raw_data_dir)
        os.unlink(path)

    data_list = process_mutag(raw_data_dir)
    torch.save(collate_data(data_list), data_filename)

    graphs = data_process(data_list)
    train_dataset, val_dataset, test_dataset, max_num_nodes, feat_dim, assign_feat_dim = prepare_data(graphs, args)


    if os.path.isfile(model_filename):
        model = GcnEncoderGraph(args.input_dim,
                                args.hidden_dim,
                                args.output_dim,
                                args.num_classes,
                                args.num_gc_layers, args=args)
    else:
        model = GcnEncoderGraph(args.input_dim,
                                args.hidden_dim,
                                args.output_dim,
                                args.num_classes,
                                args.num_gc_layers, args=args)
        train(model, train_dataset, val_dataset, test_dataset, device, args)
        model.eval()
        results_train = evaluate(train_dataset, model, args, name="Train", max_num_examples=100)
        results_test = evaluate(test_dataset, model, args, name="Test", max_num_examples=100)
        torch.save(
                    {
                        "model_type": 'gcn',
                        "results_train": results_train,
                        "results_test": results_test,
                        "model_state": model.state_dict()
                    },
                    str(model_filename),
                )

    ckpt = torch.load(model_filename, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print("__gnn_train_scores: ", ckpt['results_train'])
    print("__gnn_test_scores: ", ckpt['results_test'])

    infos = {"dataset": args.dataset, "explainer": args.explainer_name, "number_of_edges": data.edge_index.size(1), "mask_sparsity_init": get_sparsity(edge_masks), "non_zero_values_init": get_size(edge_masks), 
            "threshold": args.threshold, "num_test_nodes": args.num_test_nodes,
             "groundtruth target": is_true_label, "time": float(format(np.mean(Time), '.4f'))}
    print("__infos:" + json.dumps(infos))
    
    new_dataset = gen_dataloader(graphs)

    edge_masks_set = compute_edge_masks(args.explainer_name, new_dataset, model, device)
    edge_index_set = get_edge_index_set(new_dataset)

    related_preds = eval_related_pred_batch(model, new_dataset, edge_index_set, edge_masks_set, device)

    fidelity = eval_fidelity(related_preds)
    fidelity['mask_sparsity'] = related_preds['mask_sparsity'].mean().item()
    fidelity['expl_edges'] = related_preds['expl_edges'].mean().item()
    print("__fidelity:" + json.dumps(fidelity))


if __name__ == '__main__':
    args = arg_parse()
    main(args)