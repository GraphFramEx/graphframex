import os
import time
import sklearn.metrics
import torch
import pickle
import numpy as np
import pandas as pd
import warnings
import json
from evaluate.fidelity import (
    fidelity_acc,
    fidelity_acc_inv,
    fidelity_gnn_acc,
    fidelity_gnn_acc_inv,
    fidelity_gnn_prob,
    fidelity_gnn_prob_inv,
    fidelity_prob,
    fidelity_prob_inv,
)
import collections
import random
from gnn.model import get_gnnNets
from train_gnn import TrainModel
from gendata import get_dataset
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_data_args,
    get_graph_size_args,
)
from utils.io_utils import check_dir
from utils.gen_utils import list_to_dict
from dataset.mutag_utils.gengroundtruth import get_ground_truth_mol
from dataset.syn_utils.gengroundtruth import get_ground_truth_syn
from evaluate.accuracy import (
    get_explanation_syn,
    get_scores,
)
from evaluate.mask_utils import mask_to_shape, clean, control_sparsity, get_mask_properties
from explainer.node_explainer import *
from explainer.graph_explainer import *
from pathlib import Path
from torch_geometric.data import DataLoader


class Baseline(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        params
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.data = dataset.data
        self.dataset_name = params["dataset_name"]
        self.device = device

        self.params = params
        self.graph_classification = eval(params["graph_classification"])
        self.task = "_graph" if self.graph_classification else "_node"

        self.explainer_name = params["explainer_name"]
        self.explained_target = params["explained_target"]

        self.focus = params["focus"]


    def _eval_fid(self, related_preds):
        if self.focus == "phenomenon":
            fidelity_scores = {
                "fidelity_acc+": fidelity_acc(related_preds),
                "fidelity_acc-": fidelity_acc_inv(related_preds),
                "fidelity_prob+": fidelity_prob(related_preds),
                "fidelity_prob-": fidelity_prob_inv(related_preds),
            }
        else:
            fidelity_scores = {
                "fidelity_gnn_acc+": fidelity_gnn_acc(related_preds),
                "fidelity_gnn_acc-": fidelity_gnn_acc_inv(related_preds),
                "fidelity_gnn_prob+": fidelity_gnn_prob(related_preds),
                "fidelity_gnn_prob-": fidelity_gnn_prob_inv(related_preds),
            }
        return fidelity_scores

    def eval(self):
        related_preds = eval("self.related_pred" + self.task)()
        fidelity_scores = self._eval_fid(related_preds)
        return fidelity_scores

    def related_pred_graph(self):
        related_preds = []
        for data in self.dataset:
            if data.get('edge_mask', None) is None:
                print("No groundtruth edge mask available for this graph")
                continue
            elif (self.explained_target is not None) and (data.y != self.explained_target):
                print("The graph is not of the target class and its edge mask is: ", data.edge_mask)
                continue
            data = data.to(self.device)
            explained_y_idx = data.idx
            data.batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
            ori_prob_idx = self.model.get_prob(data).cpu().detach().numpy()[0]
            
            masked_data, maskout_data = data.clone(), data.clone()
            edge_mask = torch.Tensor(np.array(data.edge_mask).astype(float)).to(self.device)
            masked_data.edge_index = data.edge_index[:, edge_mask > 0].to(
                self.device
            )
            masked_data.edge_attr = data.edge_attr[edge_mask > 0].to(
                self.device
            )
            maskout_data.edge_index = data.edge_index[:, edge_mask <= 0].to(
                self.device
            )
            maskout_data.edge_attr = data.edge_attr[edge_mask <= 0].to(
                self.device
            )
            """elif self.mask_nature == "hard_full":
                new_edge_mask = torch.where(edge_mask > 0, 1, 0).to(self.device).long()
                masked_data.edge_weight = new_edge_mask
                maskout_data.edge_weight = 1 - new_edge_mask"""

            masked_prob_idx = self.model.get_prob(masked_data).cpu().detach().numpy()[0]
            maskout_prob_idx = self.model.get_prob(maskout_data).cpu().detach().numpy()[0]

            true_label = data.y.cpu().item()
            pred_label = np.argmax(ori_prob_idx)
            
            # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."\
            related_preds.append(
                {
                    "explained_y_idx": explained_y_idx,
                    "masked": masked_prob_idx,
                    "maskout": maskout_prob_idx,
                    "origin": ori_prob_idx,
                    "true_label": true_label,
                    "pred_label": pred_label,
                }
            )

        related_preds = list_to_dict(related_preds)
        self.num_true_expl = len(related_preds["true_label"])
        return related_preds

    def related_pred_node(self, edge_masks, node_feat_masks):
        raise NotImplementedError


def baseline_main(dataset, model, device, args):
    args.dataset = dataset
    baseline = Baseline(
        model=model,
        dataset=dataset,
        device=device,
        params=vars(args)
    )
    
    infos = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "datatype": args.datatype,
        "focus": args.focus,
        "device": str(device),
    }

    # Evaluate scores of the masks
    eval_scores = baseline.eval()
    scores = {
        key: value
        for key, value in sorted(
            infos.items()
            | eval_scores.items()
        )
    }
    results = pd.DataFrame({k: [v] for k, v in scores.items()})
    print('results', results)
    ### Save results ###
    save_path = os.path.join(
        args.result_save_dir+'_baseline', args.dataset_name
    )
    os.makedirs(save_path, exist_ok=True)
    results.to_csv(
        os.path.join(
            save_path,
            "results_{}_{}_{}_{}_target{}_{}.csv".format(
                args.dataset_name,
                args.model_name,
                args.focus,
                baseline.num_true_expl,
                args.explained_target,
                str(device)
            ),
        )
    )

if __name__=='__main__':

    parser, args = arg_parse()
    args = get_graph_size_args(args)

    if args.dataset_name.lower() in ["mutag_large", "mnist", "ba_2motifs"]:
         (
            args.groundtruth,
            args.graph_classification,
            args.num_layers,
            args.hidden_dim,
            args.num_epochs,
            args.lr,
            args.weight_decay,
            args.dropout,
            args.readout,
            args.batch_size,
            args.gamma,
            args.milestones,
            args.num_early_stop,
            args.unseen
        ) = ("True", "True", 3, 32, 200, 0.001, 5e-4, 0.0, "max", 64, 0.5, [70, 90, 120, 170], 50, "True")


    args_group = create_args_group(parser, args)

    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_params = args_group["dataset_params"]
    model_params = args_group["model_params"]

    dataset = get_dataset(
        dataset_root=args.data_save_dir,
        **dataset_params,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    args = get_data_args(dataset, args)
    dataset_params["num_classes"] = args.num_classes
    dataset_params["num_node_features"] =args.num_node_features

    print("Number of graphs: {}".format(len(dataset)))
    for k in range(args.num_classes):
        print("Number of graphs labeled as {}: {}".format(k, (dataset.data.y == k).sum().item()))
    
    if len(dataset) > 1:
        dataset_params["max_num_nodes"] = max([d.num_nodes for d in dataset])
    else:
        dataset_params["max_num_nodes"] = dataset.data.num_nodes
    args.max_num_nodes = dataset_params["max_num_nodes"]
    model_params["edge_dim"] = args.edge_dim

    
    if eval(args.graph_classification):
        args.data_split_ratio = [args.train_ratio, args.val_ratio, args.test_ratio]
        dataloader_params = {
            "batch_size": args.batch_size,
            "random_split_flag": eval(args.random_split_flag),
            "data_split_ratio": args.data_split_ratio,
            "seed": args.seed,
        }
    model = get_gnnNets(
        dataset_params["num_node_features"], dataset_params["num_classes"], model_params
    )
    model_save_name = f"{args.model_name}_{args.num_layers}l_{str(device)}"
    if args.dataset_name.startswith(tuple(["uk", "ieee"])):
        model_save_name = f"{args.datatype}_" + model_save_name
    if eval(args.graph_classification):
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=model_save_name,
            dataloader_params=dataloader_params,
        )
    else:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=model_save_name,
        )
    if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
        trainer.load_model()
    else:
        trainer.train(
            train_params=args_group["train_params"],
            optimizer_params=args_group["optimizer_params"],
        )
    _, _, _, _, _ = trainer.test()
    
    baseline_main(dataset, model, device, args)