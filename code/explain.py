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
from utils.io_utils import check_dir
from utils.gen_utils import list_to_dict
from dataset.syn_utils.gengroundtruth import get_ground_truth_syn
from evaluate.accuracy import (
    get_explanation_syn,
    get_scores,
)
from evaluate.mask_utils import (
    mask_to_shape,
    clean,
    control_sparsity,
    get_mask_properties,
)
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
        if self.dataset_name.startswith(["ba", "tree"]):
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
            if (self.dataset_name.startswith(tuple(["ba", "tree"]))) & (
                not self.graph_classification
            ):
                G_true, role, true_edge_mask = get_ground_truth_syn(
                    self.explained_y[i], self.data, self.dataset_name
                )
                G_expl = get_explanation_syn(
                    graph, edge_mask, num_top_edges=self.num_top_edges, top_acc=True
                )
                top_recall, top_precision, top_f1_score = get_scores(G_expl, G_true)
                top_balanced_acc, top_roc_auc_score = np.nan, np.nan
            elif self.dataset_name == "ba_multishapes":
                (
                    top_f1_score,
                    top_recall,
                    top_precision,
                    top_balanced_acc,
                    top_roc_auc_score,
                ) = (0, 0, 0, 0, 0)
                edge_mask = edge_mask.cpu().numpy()

                if graph.get("edge_label", None) is None:
                    print(
                        f"No true explanation available for this graph {graph.idx} with label {graph.y}."
                    )
                else:
                    edge_label = graph.edge_label.cpu().detach().numpy()
                    masked_label = edge_label[edge_mask > 0]
                    y = graph.y.item()
                    n_labels = 4
                    d = dict()
                    for i in range(1, n_labels):
                        d[i] = list(masked_label).count(i)
                    label_sum = np.where(np.array(list(d.values())) > 0, 1, 0).sum()

                if (y == 1 and label_sum == 2) or (y == 0 and label_sum in [1, 3]):
                    true_explanation = np.where(edge_label > 0, 1, 0)
                    pred_explanation = edge_mask
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        pred_explanation = np.zeros(len(edge_mask))
                        mask = edge_mask.copy()
                        if eval(self.directed):
                            unimportant_indices = (-mask).argsort()[n + 1 :]
                            mask[unimportant_indices] = 0
                        else:
                            mask = mask_to_shape(mask, graph.edge_index, n)
                        top_roc_auc_score = sklearn.metrics.roc_auc_score(
                            true_explanation, mask
                        )
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
                        top_balanced_acc = sklearn.metrics.balanced_accuracy_score(
                            true_explanation, pred_explanation
                        )
                    elif y == 0 and label_sum == 0:
                        top_roc_auc_score = np.nan
                        top_precision = np.nan
                        top_recall = np.nan
                        top_f1_score = np.nan
                        top_balanced_acc = np.nan
            elif self.dataset_name.startswith(
                tuple(
                    [
                        "uk",
                        "ieee24",
                        "ieee39",
                        "ieee118",
                        "ba_2motifs",
                        "ba_house_grid",
                        "mutag",
                        "benzene",
                        "mnist",
                    ]
                )
            ):
                (
                    top_f1_score,
                    top_recall,
                    top_precision,
                    top_balanced_acc,
                    top_roc_auc_score,
                ) = (np.nan, np.nan, np.nan, np.nan, np.nan)
                edge_mask = edge_mask.cpu().numpy()
                if graph.edge_mask is not None:
                    true_explanation = graph.edge_mask.cpu().numpy()
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        pred_explanation = np.zeros(len(edge_mask))
                        mask = edge_mask.copy()
                        if eval(self.directed):
                            unimportant_indices = (-mask).argsort()[n + 1 :]
                            mask[unimportant_indices] = 0
                        else:
                            mask = mask_to_shape(mask, graph.edge_index, n)
                        top_roc_auc_score = sklearn.metrics.roc_auc_score(
                            true_explanation, mask
                        )
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
                        top_balanced_acc = sklearn.metrics.balanced_accuracy_score(
                            true_explanation, pred_explanation
                        )
            else:
                raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
            entry = {
                "top_roc_auc_score": top_roc_auc_score,
                "top_recall": top_recall,
                "top_precision": top_precision,
                "top_f1_score": top_f1_score,
                "top_balanced_acc": top_balanced_acc,
            }
            scores.append(entry)
        accuracy_scores = pd.DataFrame.from_dict(list_to_dict(scores))
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
            if (self.dataset_name.startswith(tuple(["ba", "tree"]))) & (
                not self.graph_classification
            ):
                G_true, role, true_edge_mask = get_ground_truth_syn(
                    self.explained_y[i], self.data, self.dataset_name
                )
                G_expl = get_explanation_syn(
                    graph, edge_mask, num_top_edges=self.num_top_edges, top_acc=False
                )
                recall, precision, f1_score = get_scores(G_expl, G_true)
                balanced_acc, roc_auc_score = np.nan, np.nan
                num_explained_y_with_acc += 1

            elif self.dataset_name == "ba_multishapes":
                f1_score, recall, precision, balanced_acc, roc_auc_score = 0, 0, 0, 0, 0
                edge_mask = edge_mask.cpu().numpy()

                if graph.get("edge_label", None) is None:
                    print(
                        f"No true explanation available for this graph {graph.idx} with label {graph.y}."
                    )
                else:
                    edge_label = graph.edge_label.cpu().detach().numpy()
                    masked_label = edge_label[edge_mask > 0]
                    y = graph.y.item()
                    n_labels = 4
                    d = dict()
                    for i in range(1, n_labels):
                        d[i] = list(masked_label).count(i)
                    label_sum = np.where(np.array(list(d.values())) > 0, 1, 0).sum()

                if (y == 1 and label_sum == 2) or (y == 0 and label_sum in [0, 1, 3]):
                    true_explanation = np.where(edge_label > 0, 1, 0)
                    pred_explanation = edge_mask
                    roc_auc_score = sklearn.metrics.roc_auc_score(
                        true_explanation, edge_mask
                    )
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
                    balanced_acc = sklearn.metrics.balanced_accuracy_score(
                        true_explanation, pred_explanation
                    )
                    num_explained_y_with_acc += 1

            elif self.dataset_name.startswith(
                tuple(
                    [
                        "uk",
                        "ieee24",
                        "ieee39",
                        "ieee118",
                        "ba_2motifs",
                        "ba_house_grid",
                        "mutag",
                        "benzene",
                        "mnist",
                    ]
                )
            ):
                f1_score, recall, precision, balanced_acc, roc_auc_score = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
                edge_mask = edge_mask.cpu().numpy()
                if graph.get("edge_mask", None) is None:
                    print(
                        f"No true explanation available for this graph {graph.idx} with label {graph.y}."
                    )
                else:
                    true_explanation = graph.edge_mask.cpu().numpy()
                    n = len(np.where(true_explanation == 1)[0])
                    if n > 0:
                        roc_auc_score = sklearn.metrics.roc_auc_score(
                            true_explanation, edge_mask
                        )
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
                        balanced_acc = sklearn.metrics.balanced_accuracy_score(
                            true_explanation, pred_explanation
                        )
                        num_explained_y_with_acc += 1
            else:
                raise ValueError("Unknown dataset name: {}".format(self.dataset_name))
            entry = {
                "roc_auc_score": roc_auc_score,
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score,
                "balanced_acc": balanced_acc,
            }
            scores.append(entry)
        accuracy_scores = pd.DataFrame.from_dict(list_to_dict(scores))
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
        elif self.focus == "model":
            fidelity_scores = {
                "fidelity_gnn_acc+": fidelity_gnn_acc(related_preds),
                "fidelity_gnn_acc-": fidelity_gnn_acc_inv(related_preds),
                "fidelity_gnn_prob+": fidelity_gnn_prob(related_preds),
                "fidelity_gnn_prob-": fidelity_gnn_prob_inv(related_preds),
            }
        else:
            raise ValueError("Unknown focus: {}".format(self.focus))

        fidelity_scores = pd.DataFrame.from_dict(fidelity_scores)
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
        return (
            top_accuracy_scores,
            accuracy_scores,
            fidelity_scores,
        )

    def related_pred_graph(self, edge_masks, node_feat_masks):
        related_preds = []
        for i in range(len(self.explained_y)):
            explained_y_idx = self.explained_y[i]
            data = self.dataset[explained_y_idx]
            data = data.to(self.device)
            data.batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
            ori_prob_idx = self.model.get_prob(data).cpu().detach().numpy()[0]
            if node_feat_masks[0] is not None:
                if node_feat_masks[i].ndim == 0:
                    # if type of node feat mask is 'feature'
                    node_feat_mask = node_feat_masks[i].reshape(-1)
                else:
                    # if type of node feat mask is 'feature'
                    node_feat_mask = node_feat_masks[i]
                node_feat_mask = torch.Tensor(node_feat_mask).to(self.device)
                x_masked = data.x * node_feat_mask
                x_maskout = data.x * (1 - node_feat_mask)
            else:
                x_masked, x_maskout = data.x, data.x

            masked_data, maskout_data = (
                data.clone(),
                data.clone(),
            )
            masked_data.x, maskout_data.x = (x_masked, x_maskout)

            if (
                (edge_masks[i] is not None)
                and (hasattr(edge_masks[i], "__len__"))
                and (len(edge_masks[i]) > 0)
            ):
                edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
                hard_edge_mask = (
                    torch.where(edge_mask > 0, 1, 0).to(self.device).float()
                )
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
                elif self.mask_nature == "hard_full":
                    masked_data.edge_weight = hard_edge_mask
                    maskout_data.edge_weight = 1 - hard_edge_mask
                elif self.mask_nature == "soft":
                    masked_data.edge_weight = edge_mask
                    maskout_data.edge_weight = 1 - edge_mask
                else:
                    raise ValueError("Unknown mask nature: {}".format(self.mask_nature))

            masked_prob_idx = self.model.get_prob(masked_data).cpu().detach().numpy()[0]
            maskout_prob_idx = (
                self.model.get_prob(maskout_data).cpu().detach().numpy()[0]
            )

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
        return related_preds

    def related_pred_node(self, edge_masks, node_feat_masks):
        related_preds = []
        data = self.data
        ori_probs = self.model.get_prob(data=self.data)
        for i in range(len(self.explained_y)):
            if node_feat_masks[0] is not None:
                if node_feat_masks[i].ndim == 0:
                    # if type of node feat mask is 'feature'
                    node_feat_mask = node_feat_masks[i].reshape(-1)
                else:
                    # if type of node feat mask is 'feature'
                    node_feat_mask = node_feat_masks[i]
                node_feat_mask = torch.Tensor(node_feat_mask).to(self.device)
                x_masked = self.data.x * node_feat_mask
                x_maskout = self.data.x * (1 - node_feat_mask)
            else:
                x_masked, x_maskout = self.data.x, self.data.x

            masked_data, maskout_data = (
                data.clone(),
                data.clone(),
            )
            masked_data.x, maskout_data.x.x = (
                x_masked,
                x_maskout,
            )

            if (
                (edge_masks[i] is not None)
                and (hasattr(edge_masks[i], "__len__"))
                and (len(edge_masks[i]) > 0)
            ):
                edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
                hard_edge_mask = (
                    torch.where(edge_mask > 0, 1, 0).to(self.device).float()
                )
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
                elif self.mask_nature == "hard_full":
                    masked_data.edge_weight = hard_edge_mask
                    maskout_data.edge_weight = 1 - hard_edge_mask
                elif self.mask_nature == "soft":
                    masked_data.edge_weight = edge_mask
                    maskout_data.edge_weight = 1 - edge_mask
                else:
                    raise ValueError("Unknown mask nature: {}".format(self.mask_nature))

            masked_probs = self.model.get_prob(masked_data).cpu().detach().numpy()[0]
            maskout_probs = self.model.get_prob(maskout_data).cpu().detach().numpy()[0]

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
            **self.explainer_params,
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
        if (self.save_dir is not None) and (
            Path(os.path.join(self.save_dir, self.save_name)).is_file()
        ):
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
                if (
                    (edge_mask is not None)
                    and (hasattr(edge_mask, "__len__"))
                    and (len(edge_mask) > 0)
                ):
                    edge_masks.append(edge_mask)
                    node_feat_masks.append(node_feat_mask)
                    computation_time.append(duration_seconds)
                    final_explained_y.append(explained_y_idx)
            self.explained_y = final_explained_y
            if (self.save_dir is not None) and self.save:
                self.save_mask(
                    final_explained_y, edge_masks, node_feat_masks, computation_time
                )
        return self.explained_y, edge_masks, node_feat_masks, computation_time

    def clean_mask(self, edge_masks, node_feat_masks):
        if edge_masks:
            if edge_masks[0] is not None:
                edge_masks = clean(edge_masks)
        if node_feat_masks:
            node_feat_masks = np.array(node_feat_masks)
            if node_feat_masks[0] is not None:
                node_feat_masks = clean(node_feat_masks)
        return edge_masks, node_feat_masks

    def _transform(self, masks, param):
        """Transform masks according to the given strategy (topk, threshold, sparsity) and level."""
        new_masks = []
        if (param is None) | (self.mask_transformation == "None"):
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
                    unimportant_indices = (-mask).argsort()[param + 1 :]
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
                list_idx = np.array(self.list_test_idx)[
                    np.where(pred_labels == true_labels)[0].astype(int)
                ]
            elif self.pred_type == "wrong":
                list_idx = np.array(self.list_test_idx)[
                    np.where(pred_labels != true_labels)[0].astype(int)
                ]
            elif self.pred_type == "mix":
                list_idx = np.array(self.list_test_idx)
            else:
                raise ValueError("pred_type must be correct, wrong or mix.")
            # list_idx = self.list_test_idx
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
                list_idx = np.array(self.list_test_idx)
            else:
                raise ValueError("pred_type must be correct, wrong or mix.")
            print("Number of explanable entities: ", len(list_idx))
            explained_y = np.random.choice(
                list_idx,
                size=min(self.num_explained_y, len(list_idx), self.data.num_nodes),
                replace=False,
            )
        print("Number of explained entities: ", len(explained_y))
        return explained_y

    def save_mask(self, explained_y, edge_masks, node_feat_masks, computation_time):
        assert self.save_dir is not None, "save_dir is None. Masks are not saved"
        save_path = os.path.join(self.save_dir, self.save_name)
        with open(save_path, "wb") as f:
            pickle.dump([explained_y, edge_masks, node_feat_masks, computation_time], f)

    def load_mask(self):
        assert self.save_dir is not None, "save_dir is None. No mask to be loaded"
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
        args.seed,
    )
    return mask_save_name


def avg_scores(scores):
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            avg_scores = {k: np.nanmean(v) for k, v in scores.items()}
        except RuntimeWarning:
            avg_scores = {}
    return avg_scores


def explain_main(dataset, model, device, args, unseen=False):
    args.dataset = dataset
    if unseen:
        args.pred_type = "mix"
        # args.mask_save_dir="None"
        args.num_explained_y = len(dataset)
    mask_save_name = get_mask_dir_path(args, device, unseen)

    if (args.explained_target is None) | (unseen):
        list_test_idx = range(0, len(dataset.data.y))
    else:
        list_test_idx = np.where(dataset.data.y.cpu().numpy() == args.explained_target)[
            0
        ]
    print("Number of explanable entities: ", len(list_test_idx))
    explainer = Explain(
        model=model,
        dataset=dataset,
        device=device,
        list_test_idx=list_test_idx,
        explainer_params=vars(args),
        save_dir=None
        if args.mask_save_dir == "None"
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
        "explained_target": args.explained_target,
        "pred_type": args.pred_type,
        "time": float(format(np.mean(computation_time), ".4f")),
        "device": str(device),
    }

    if (edge_masks is None) or (not edge_masks):
        raise ValueError("Edge masks are None")
    params_lst = eval(explainer.transf_params)
    params_lst.insert(0, None)
    edge_masks_ori = edge_masks.copy()
    for i, param in enumerate(params_lst):
        params_transf = {explainer.mask_transformation: param}
        edge_masks = explainer._transform(edge_masks_ori, param)
        # Compute mask properties
        edge_masks_properties = get_mask_properties(edge_masks)
        # Evaluate scores of the masks
        (
            top_accuracy_scores,
            accuracy_scores,
            fidelity_scores,
        ) = explainer.eval(edge_masks, node_feat_masks)
        eval_scores = {
            **top_accuracy_scores,
            **accuracy_scores,
            **fidelity_scores,
        }
        scores = pd.DataFrame.from_dict(eval_scores)
        for column_name, values in {**infos, **params_transf}.items():
            scores[column_name] = values
        if i == 0:
            results = scores
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
                args.seed,
            ),
        )
    )
