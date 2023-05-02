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


class Explain(object):
    def __init__(
        self,
        model,
        dataset,
        device,
        list_test_idx,
        explainer_params,
        save_dir=None,
        save_name="mask",
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.data = dataset.data
        self.dataset_name = explainer_params["dataset_name"]
        self.device = device
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        if self.save_dir is not None:
            check_dir(self.save_dir)

        self.explainer_params = explainer_params
        self.graph_classification = eval(explainer_params["graph_classification"])
        self.task = "_graph" if self.graph_classification else "_node"

        self.list_test_idx = list_test_idx
        self.explainer_name = explainer_params["explainer_name"]
        self.num_explained_y = explainer_params["num_explained_y"]
        self.explained_target = explainer_params["explained_target"]
        self.pred_type = explainer_params["pred_type"]

        self.focus = explainer_params["focus"]
        self.mask_nature = explainer_params["mask_nature"]
        self.mask_transformation = explainer_params["mask_transformation"]
        self.transf_params = explainer_params["transf_params"]
        self.directed = explainer_params["directed"]
        self.groundtruth = eval(explainer_params["groundtruth"])
        if self.groundtruth:
            self.num_top_edges = explainer_params["num_top_edges"]

    def get_ground_truth(self, **kwargs):
        if self.dataset_name == "mutag":
            G_true_list = get_ground_truth_mol(self.dataset_name)
        elif self.dataset_name.startswith(["ba", "tree"]):
            G_true, role, true_edge_mask = get_ground_truth_syn(
                kwargs["explained_idx"], self.data, self.dataset_name
            )
            G_true_list = [G_true]
        else:
            raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
        return G_true_list

    def _eval_top_acc(self, edge_masks):
        print("Top Accuracy is being computed...")
        scores = []
        for i in range(len(self.explained_y)):
            edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
            graph = (
                self.dataset[self.explained_y[i]]
                if self.graph_classification
                else self.dataset.data
            )
            if (self.dataset_name.startswith(tuple(["ba", "tree"]))) & (self.dataset_name != "ba_2motifs"):
                G_true, role, true_edge_mask = get_ground_truth_syn(
                    self.explained_y[i], self.data, self.dataset_name
                )
                G_expl = get_explanation_syn(
                    graph, edge_mask, num_top_edges=self.num_top_edges, top_acc=True
                )
                top_recall, top_precision, top_f1_score = get_scores(G_expl, G_true)
                top_balanced_acc = None
            elif self.dataset_name.startswith(tuple(["uk", "ieee24", "ieee39", "ieee118", "ba_2motifs"])):
                top_f1_score, top_recall, top_precision, top_balanced_acc, top_roc_auc_score = np.nan, np.nan, np.nan, np.nan, np.nan
                edge_mask = edge_mask.cpu().numpy()
                if graph.edge_mask is not None:
                    true_explanation = graph.edge_mask.cpu().numpy()
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        pred_explanation = np.zeros(len(edge_mask))
                        mask = edge_mask.copy()
                        if eval(self.directed):
                            unimportant_indices = (-mask).argsort()[n+1:]
                            mask[unimportant_indices] = 0
                        else:
                            mask = mask_to_shape(mask, graph.edge_index, n)
                        top_roc_auc_score = sklearn.metrics.roc_auc_score(true_explanation, mask)
                        pred_explanation[mask > 0] = 1
                        top_precision = sklearn.metrics.precision_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        top_recall = sklearn.metrics.recall_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        top_f1_score = sklearn.metrics.f1_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        top_balanced_acc = sklearn.metrics.balanced_accuracy_score(true_explanation, pred_explanation)
            else:
                raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
            entry = {"top_roc_auc_score":top_roc_auc_score,"top_recall": top_recall, "top_precision": top_precision, "top_f1_score": top_f1_score, "top_balanced_acc": top_balanced_acc}
            scores.append(entry)
        scores = list_to_dict(scores)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                accuracy_scores = {k: np.nanmean(v) for k, v in scores.items()}
            except RuntimeWarning:
                accuracy_scores = {}
        return accuracy_scores
            

    def _eval_acc(self, edge_masks):
        scores = []
        num_explained_y_with_acc = 0
        for i in range(len(self.explained_y)):
            edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
            graph = (
                self.dataset[self.explained_y[i]]
                if self.graph_classification
                else self.dataset.data
            )
            if (self.dataset_name.startswith(tuple(["ba", "tree"]))) & (self.dataset_name != "ba_2motifs"):
                G_true, role, true_edge_mask = get_ground_truth_syn(
                    self.explained_y[i], self.data, self.dataset_name
                )
                G_expl = get_explanation_syn(
                    graph, edge_mask, num_top_edges=self.num_top_edges, top_acc=False
                )
                recall, precision, f1_score = get_scores(G_expl, G_true)
                num_explained_y_with_acc += 1
            elif self.dataset_name.startswith(tuple(["uk", "ieee24", "ieee39", "ieee118", "ba_2motifs"])):
                f1_score, recall, precision, balanced_acc, roc_auc_score = np.nan, np.nan, np.nan, np.nan, np.nan
                edge_mask = edge_mask.cpu().numpy()
                if graph.edge_mask is not None:
                    true_explanation = graph.edge_mask.cpu().numpy()
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        roc_auc_score = sklearn.metrics.roc_auc_score(true_explanation, edge_mask)
                        pred_explanation = np.zeros(len(edge_mask))
                        pred_explanation[edge_mask > 0] = 1
                        precision = sklearn.metrics.precision_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        recall = sklearn.metrics.recall_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        f1_score = sklearn.metrics.f1_score(
                            true_explanation, pred_explanation, pos_label=1
                        )
                        balanced_acc = sklearn.metrics.balanced_accuracy_score(true_explanation, pred_explanation)
                        num_explained_y_with_acc += 1
            else:
                raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
            entry = {"roc_auc_score":roc_auc_score, "recall": recall, "precision": precision, "f1_score": f1_score, "balanced_acc": balanced_acc}
            scores.append(entry)
        scores = list_to_dict(scores)
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                accuracy_scores = {k: np.nanmean(v) for k, v in scores.items()}
            except RuntimeWarning:
                accuracy_scores = {}
        accuracy_scores["num_explained_y_with_acc"] = num_explained_y_with_acc
        return accuracy_scores

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
        fidelity_scores["num_explained_y_fid"] = self.num_explained_y
        return fidelity_scores

    def eval(self, edge_masks, node_feat_masks):
        related_preds = eval("self.related_pred" + self.task)(
            edge_masks, node_feat_masks
        )
        if self.groundtruth:
            accuracy_scores = self._eval_acc(edge_masks)
            top_accuracy_scores = self._eval_top_acc(edge_masks)
        else:
            accuracy_scores, top_accuracy_scores = {}, {}
        fidelity_scores = self._eval_fid(related_preds)
        return top_accuracy_scores, accuracy_scores, fidelity_scores

    def related_pred_graph(self, edge_masks, node_feat_masks):
        print("Computing related predictions for graph classification")
        related_preds = []
        for i in range(len(self.explained_y)):
            explained_y_idx = self.explained_y[i]
            data = self.dataset[explained_y_idx]
            data = data.to(self.device)
            data.batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
            ori_prob_idx = self.model.get_prob(data).cpu().detach().numpy()[0]
            if node_feat_masks[0] is not None:
                node_feat_mask = torch.Tensor(node_feat_masks[i]).to(self.device)
                if node_feat_mask.dim() == 2:
                    x_masked = node_feat_mask
                    x_maskout = 1 - node_feat_mask
                else:
                    x_masked = data.x * node_feat_mask
                    x_maskout = data.x * (1 - node_feat_mask)
            else:
                x_masked, x_maskout = data.x, data.x

            masked_data, maskout_data = data.clone(), data.clone()
            masked_data.x, maskout_data.x = x_masked, x_maskout

            if edge_masks[0] is not None:
                edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
                if self.mask_nature == "hard":
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
                elif self.mask_nature == "soft_binary":
                    new_edge_mask = torch.where(edge_mask > 0, 1, 0).to(self.device).long()
                    masked_data.edge_weight = new_edge_mask
                    maskout_data.edge_weight = 1 - new_edge_mask
                elif self.mask_nature == "soft":
                    new_edge_mask = edge_mask
                    masked_data.edge_weight = new_edge_mask
                    maskout_data.edge_weight = 1 - new_edge_mask
                else:
                    raise ValueError("Unknown mask nature: {}".format(self.mask_nature))

            masked_prob_idx = self.model.get_prob(masked_data).cpu().detach().numpy()[0]
            maskout_prob_idx = self.model.get_prob(maskout_data).cpu().detach().numpy()[0]

            true_label = data.y.cpu().item()
            pred_label = np.argmax(ori_prob_idx)

            print('masked_prob_idx', masked_prob_idx)
            
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
        return related_preds

    def related_pred_node(self, edge_masks, node_feat_masks):
        related_preds = []
        ori_probs = self.model.get_prob(data=self.data)
        for i in range(len(self.explained_y)):
            if node_feat_masks[0] is not None:
                node_feat_mask = torch.Tensor(node_feat_masks[i]).to(self.device)
                if node_feat_mask.dim() == 2:
                    x_masked = node_feat_mask
                    x_maskout = 1 - node_feat_mask
                else:
                    x_masked = self.data.x * node_feat_mask
                    x_maskout = self.data.x * (1 - node_feat_mask)
            else:
                x_masked, x_maskout = self.data.x, self.data.x

            if edge_masks[0] is None:
                if self.mask_nature == "hard":
                    masked_probs = self.model.get_prob(x_masked, self.data.edge_index)
                    maskout_probs = self.model.get_prob(x_maskout, self.data.edge_index)
                else:
                    masked_probs = self.model.get_prob(
                        x_masked, self.data.edge_index, self.data.edge_attr
                    )
                    maskout_probs = self.model.get_prob(
                        x_maskout, self.data.edge_index, self.data.edge_attr
                    )

            else:
                edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
                if self.mask_nature == "hard":
                    masked_edge_index = self.data.edge_index[:, edge_mask > 0].to(
                        self.device
                    )
                    maskout_edge_index = self.data.edge_index[:, edge_mask <= 0].to(
                        self.device
                    )
                    masked_probs = self.model.get_prob(x_masked, masked_edge_index)
                    maskout_probs = self.model.get_prob(x_maskout, maskout_edge_index)
                else:
                    masked_probs = self.model.get_prob(
                        x_masked,
                        self.data.edge_index,
                        self.data.edge_attr * edge_mask[:, None],
                    )
                    maskout_probs = self.model.get_prob(
                        x_maskout,
                        self.data.edge_index,
                        self.data.edge_attr * (1 - edge_mask)[:, None],
                    )
                edge_mask = edge_mask.cpu().detach().numpy()

            explained_y_idx = self.explained_y[i]
            ori_prob_idx = ori_probs[explained_y_idx].cpu().detach().numpy()
            masked_prob_idx = masked_probs[explained_y_idx].cpu().detach().numpy()
            maskout_prob_idx = maskout_probs[explained_y_idx].cpu().detach().numpy()
            true_label = self.data.y[explained_y_idx].cpu().item()
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
        return related_preds

    def _compute_graph(self, explained_y_idx):
        data = self.dataset[explained_y_idx].to(self.device)
        if self.focus == "phenomenon":
            target = data.y
        else:
            target = self.model(data=data).argmax(-1).item()
        start_time = time.time()
        edge_mask, node_feat_mask = self.explain_function(
            self.model, data, target, self.device, **self.explainer_params
        )
        end_time = time.time()
        duration_seconds = end_time - start_time
        return (
            edge_mask,
            node_feat_mask,
            duration_seconds,
        )

    def _compute_node(self, explained_y_idx):
        if self.focus == "phenomenon":
            targets = self.data.y
        else:
            self.model.eval()
            data = self.data.to(self.device)
            out = self.model(data=data)
            targets = torch.LongTensor(out.argmax(dim=1).detach().cpu().numpy()).to(
                self.device
            )
        start_time = time.time()
        edge_mask, node_feat_mask = self.explain_function(
            self.model,
            self.data,
            explained_y_idx,
            targets[explained_y_idx],
            self.device,
            **self.explainer_params
        )
        end_time = time.time()
        duration_seconds = end_time - start_time
        return (
            edge_mask,
            node_feat_mask,
            duration_seconds,
        )

    def compute_mask(self):
        self.explain_function = eval("explain_" + self.explainer_name + self.task)
        print("Computing masks using " + self.explainer_name + " explainer.")
        if (self.save_dir is not None) and (Path(os.path.join(self.save_dir, self.save_name)).is_file()):
            (
                explained_y,
                edge_masks,
                node_feat_masks,
                computation_time,
            ) = self.load_mask()
            self.explained_y = explained_y
        else:
            init_explained_y = self._get_explained_y()
            final_explained_y, edge_masks, node_feat_masks, computation_time = (
                [],
                [],
                [],
                [],
            )
            for explained_y_idx in init_explained_y:
                edge_mask, node_feat_mask, duration_seconds = eval(
                    "self._compute" + self.task
                )(explained_y_idx)
                if edge_mask is not None:
                    edge_masks.append(edge_mask)
                    node_feat_masks.append(node_feat_mask)
                    computation_time.append(duration_seconds)
                    final_explained_y.append(explained_y_idx)
            self.explained_y = final_explained_y
            if self.save:
                self.save_mask(
                    final_explained_y, edge_masks, node_feat_masks, computation_time
                )
        return self.explained_y, edge_masks, node_feat_masks, computation_time

    def clean_mask(self, edge_masks, node_feat_masks):
        if edge_masks:
            if edge_masks[0] is not None:
                edge_masks = clean(edge_masks)
        if node_feat_masks:
            if node_feat_masks[0] is not None:
                node_feat_masks = clean(node_feat_masks)
        return edge_masks, node_feat_masks

    def _transform(self, masks, param):
        """Transform masks according to the given strategy (topk, threshold, sparsity) and level."""
        new_masks = []
        if (param is None) | (self.mask_transformation=="None"):
            return masks
        for i in range(len(self.explained_y)):
            mask = masks[i].copy()
            idx = self.explained_y[i]
            edge_index = (
                self.dataset[idx].edge_index
                if self.graph_classification
                else self.data.edge_index
            )
            if self.mask_transformation == "topk":
                if eval(self.directed):
                    unimportant_indices = (-mask).argsort()[param+1:]
                    mask[unimportant_indices] = 0
                else:
                    mask = mask_to_shape(mask, edge_index, param)
                    # indices = np.where(mask > 0)[0]
            if self.mask_transformation == "sparsity":
                mask = control_sparsity(mask, param)
            if self.mask_transformation == "threshold":
                mask = np.where(mask > param, mask, 0)
            new_masks.append(mask.astype("float"))
        return new_masks

    def _get_explained_y(self):
        if self.graph_classification:
            dataloader = DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                shuffle=True,
            )
            data = next(iter(dataloader)).to(self.device)
            logits = self.model(data)
            pred_labels = logits.argmax(-1).cpu().numpy()[self.list_test_idx]
            true_labels = self.dataset.data.y.cpu().numpy()[self.list_test_idx]
            if self.pred_type == "correct":
                list_idx = np.where(pred_labels == true_labels)[0]
            elif self.pred_type == "wrong":
                list_idx = np.where(pred_labels != true_labels)[0]
            elif self.pred_type == "mix":
                list_idx = self.list_test_idx
            else:
                raise ValueError("pred_type must be correct, wrong or mix.")
            explained_y = np.random.choice(
                list_idx,
                size=min(len(list_idx), self.num_explained_y, len(self.dataset)),
                replace=False,
            )
        else:
            logits = self.model(self.data)
            pred_labels = logits.argmax(-1).cpu().numpy()[self.list_test_idx]
            true_labels = self.data.y.cpu().numpy()[self.list_test_idx]
            if self.pred_type == "correct":
                list_idx = np.where(pred_labels == true_labels)[0]
            elif self.pred_type == "wrong":
                list_idx = np.where(pred_labels != true_labels)[0]
            elif self.pred_type == "mix":
                list_idx = self.list_test_idx
            else:
                raise ValueError("pred_type must be correct, wrong or mix.")
            print("Number of explanable entities: ", len(list_idx))
            explained_y = np.random.choice(
                list_idx, size=min(self.num_explained_y, len(list_idx), self.data.num_nodes), replace=False
            )
        print("Number of explained entities: ", len(explained_y))
        return explained_y

    def save_mask(self, explained_y, edge_masks, node_feat_masks, computation_time):
        if self.save_dir is None:
            print("save_dir is None. Masks are not saved")
            return
        else:
            save_path = os.path.join(self.save_dir, self.save_name)
            with open(save_path, "wb") as f:
                pickle.dump([explained_y, edge_masks, node_feat_masks, computation_time], f)

    def load_mask(self):
        save_path = os.path.join(self.save_dir, self.save_name)
        with open(save_path, "rb") as f:
            w_list = pickle.load(f)
        explained_y, edge_masks, node_feat_masks, computation_time = tuple(w_list)
        self.explained_y = explained_y
        return explained_y, edge_masks, node_feat_masks, computation_time


def get_mask_dir_path(args, device, unseen=False):
    unseen_str = "_unseen" if unseen else ""
    mask_save_name = "mask{}_{}_{}_{}_{}_{}_target{}_{}_{}_{}.pkl".format(
        unseen_str,
        args.dataset_name,
        args.model_name,
        args.explainer_name,
        args.focus,
        args.num_explained_y,
        args.explained_target,
        args.pred_type,
        str(device),
        args.seed
    )
    return mask_save_name


def explain_main(dataset, model, device, args, unseen=False):

    mask_save_name = get_mask_dir_path(args, device, unseen)
    args.dataset = dataset

    if unseen:
        args.prediction_type = "mix"
        args.mask_save_dir="None"
        args.num_explained_y = len(dataset)
    
    if (args.explained_target is None) | (unseen):
        list_test_idx = range(0, len(dataset.data.y))
    else:
        list_test_idx = np.where(dataset.data.y.cpu().numpy() == args.explained_target)[0]
    print("Number of explanable entities: ", len(list_test_idx))
    explainer = Explain(
        model=model,
        dataset=dataset,
        device=device,
        list_test_idx=list_test_idx,
        explainer_params=vars(args),
        save_dir=None
        if args.mask_save_dir=="None"
        else os.path.join(args.mask_save_dir, args.dataset_name, args.explainer_name),
        save_name=mask_save_name,
    )

    (
        explained_y,
        edge_masks,
        node_feat_masks,
        computation_time,
    ) = explainer.compute_mask()
    edge_masks, node_feat_masks = explainer.clean_mask(edge_masks, node_feat_masks)

    infos = {
        "seed": args.seed,
        "dataset": args.dataset_name,
        "model": args.model_name,
        "datatype": args.datatype,
        "explainer": args.explainer_name,
        "focus": args.focus,
        "mask_nature": args.mask_nature,
        "pred_type": args.pred_type,
        "time": float(format(np.mean(computation_time), ".4f")),
        "device": str(device),
    }

    if (edge_masks is None) or (not edge_masks):
        raise ValueError("Edge masks are None")
    params_lst = eval(explainer.transf_params)
    params_lst.insert(0, None)
    edge_masks_ori = edge_masks.copy()
    print('params list: ', params_lst)
    for i, param in enumerate(params_lst):
        params_transf = {explainer.mask_transformation: param}
        edge_masks = explainer._transform(edge_masks_ori, param)
        # Compute mask properties
        edge_masks_properties = get_mask_properties(edge_masks)
        # Evaluate scores of the masks
        top_accuracy_scores, accuracy_scores, fidelity_scores = explainer.eval(edge_masks, node_feat_masks)
        eval_scores = {**top_accuracy_scores, **accuracy_scores, **fidelity_scores}
        scores = {
            key: value
            for key, value in sorted(
                infos.items()
                | edge_masks_properties.items()
                | eval_scores.items()
                | params_transf.items()
            )
        }
        if i == 0:
            results = pd.DataFrame({k: [v] for k, v in scores.items()})
        else:
            results = results.append(scores, ignore_index=True)
    ### Save results ###
    save_path = os.path.join(
        args.result_save_dir, args.dataset_name, args.explainer_name
    )
    os.makedirs(save_path, exist_ok=True)
    unseen_str = "_unseen" if unseen else ""
    results.to_csv(
        os.path.join(
            save_path,
            "results{}_{}_{}_{}_{}_{}_{}_target{}_{}_{}_{}.csv".format(
                unseen_str,
                args.dataset_name,
                args.model_name,
                args.explainer_name,
                args.focus,
                args.mask_nature,
                args.num_explained_y,
                args.explained_target,
                args.pred_type,
                str(device),
                args.seed
            ),
        )
    )

if __name__=='__main__':

    parser, args = arg_parse()
    args = get_graph_size_args(args)

    (args.groundtruth,
        args.graph_classification,
        args.num_layers,
        args.hidden_dim,
        args.num_epochs,
        args.lr,
        args.weight_decay,
        args.dropout,
        args.readout,
        args.batch_size,
        args.unseen
    ) = ("False", "True", 3, 16, 200, 0.001, 5e-4, 0.0, "max", 64, "False")
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

    data_y = dataset.data.y.cpu().numpy()
    if args.num_classes == 2:
        y_cf_all = 1 - data_y
    else:
        y_cf_all = []
        for y in data_y:
            y_cf_all.append(y+1 if y < args.num_classes - 1 else 0)
    args.y_cf_all = torch.FloatTensor(y_cf_all).to(device)

    
    print("num_classes:", dataset_params["num_classes"])
    print("num_node_features:", dataset_params["num_node_features"])
    print("dataset length:", len(dataset))
    if len(dataset) > 1:
        dataset_params["max_num_nodes"] = max([d.num_nodes for d in dataset])
    else:
        dataset_params["max_num_nodes"] = dataset.data.num_nodes
    args.max_num_nodes = dataset_params["max_num_nodes"]
    args.edge_dim = dataset.data.edge_attr.size(1)
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
    
    mask_save_name = get_mask_dir_path(args, device, eval(args.unseen))
    args.dataset = dataset
    list_test_idx = range(0, len(dataset.data.y))
    print("Number of explanable entities: ", len(list_test_idx))
    explainer = Explain(
        model=model,
        dataset=dataset,
        device=device,
        list_test_idx=list_test_idx,
        explainer_params=vars(args),
        save_dir=None
        if args.mask_save_dir=="None"
        else os.path.join(args.mask_save_dir, args.dataset_name, args.explainer_name),
        save_name=mask_save_name,
    )

    (
        explained_y,
        edge_masks,
        node_feat_masks,
        computation_time,
    ) = explainer.compute_mask()
    edge_masks, node_feat_masks = explainer.clean_mask(edge_masks, node_feat_masks)


    dict_mask = dict(zip(explained_y,edge_masks))
    ordered_dict_mask = collections.OrderedDict(sorted(dict_mask.items()))
    d = {int(key): value.tolist() for key, value in ordered_dict_mask.items()}
    json.dump(d, open(os.path.join(args.mask_save_dir, args.dataset_name, args.explainer_name, mask_save_name.replace(".pkl", ".json")), "w"))
