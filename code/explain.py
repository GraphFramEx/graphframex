import os
import time
import torch
import pickle
import numpy as np
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
from dataset.syn_utils.gengroundtruth import get_ground_truth
from evaluate.accuracy import get_explanation, get_scores
from evaluate.mask_utils import mask_to_shape, clean, control_sparsity
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
        graph_classification,
        dataset_name,
        explainer_params,
        save_dir=None,
        save_name="mask",
    ):
        self.model = model
        self.dataset = dataset  # train_mask, eval_mask, test_mask
        self.data = dataset.data
        self.dataset_name = dataset_name
        self.device = device

        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        self.explainer_params = explainer_params
        self.graph_classification = graph_classification
        self.task = "_graph" if self.graph_classification else "_node"

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
            self.top_acc = explainer_params["top_acc"]
            self.num_top_edges = explainer_params["num_top_edges"]

    def _eval_acc(self, edge_masks):
        scores = []
        for i in range(len(self.explained_y)):
            edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
            G_true, role, true_edge_mask = get_ground_truth(
                self.explained_y[i], self.data, self.dataset_name
            )
            G_expl = get_explanation(
                self.data, edge_mask, self.top_acc, self.num_top_edges
            )
            recall, precision, f1_score = get_scores(G_expl, G_true)
            entry = {"recall": recall, "precision": precision, "f1_score": f1_score}
            scores.append(entry)
        scores = list_to_dict(scores)
        accuracy_scores = {k: np.mean(v) for k, v in scores.items()}
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
        return fidelity_scores

    def eval(self, edge_masks, node_feat_masks):
        related_preds = eval("self.related_pred" + self.task)(
            edge_masks, node_feat_masks
        )
        if self.groundtruth:
            accuracy_scores = self._eval_acc(edge_masks)
        else:
            accuracy_scores = None
        fidelity_scores = self._eval_fid(related_preds)
        return accuracy_scores, fidelity_scores

    def related_pred_graph(self, edge_masks, node_feat_masks):
        related_preds = []
        for i in range(self.num_explained_y):
            explained_y_idx = self.explained_y[i]
            data = self.dataset[explained_y_idx]
            data.batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
            ori_prob_idx = self.model(data).cpu().detach().numpy()[0]
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

            if edge_masks[0] is None:
                if self.mask_nature == "hard":
                    masked_prob_idx = (
                        self.model(x_masked, data.edge_index).cpu().detach().numpy()[0]
                    )
                    maskout_prob_idx = (
                        self.model(x_maskout, data.edge_index).cpu().detach().numpy()[0]
                    )
                else:
                    masked_prob_idx = (
                        self.model(x_masked, data.edge_index, data.edge_attr)
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                    maskout_prob_idx = (
                        self.model(x_maskout, data.edge_index, data.edge_attr)
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )

            else:
                edge_mask = torch.Tensor(edge_masks[i]).to(self.device)
                if self.mask_nature == "hard":
                    masked_edge_index = data.edge_index[:, edge_mask > 0].to(
                        self.device
                    )
                    maskout_edge_index = data.edge_index[:, edge_mask <= 0].to(
                        self.device
                    )
                    masked_prob_idx = (
                        self.model(x_masked, masked_edge_index)
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                    maskout_prob_idx = (
                        self.model(x_maskout, maskout_edge_index)
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                else:
                    masked_prob_idx = (
                        self.model(
                            x_masked,
                            data.edge_index,
                            data.edge_attr * edge_mask,
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                    maskout_prob_idx = (
                        self.model(
                            x_maskout,
                            data.edge_index,
                            data.edge_attr * (1 - edge_mask),
                        )
                        .cpu()
                        .detach()
                        .numpy()[0]
                    )
                edge_mask = edge_mask.cpu().detach().numpy()

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
        ori_probs = self.model(data=self.data)
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
                    masked_probs = self.model(x_masked, self.data.edge_index)
                    maskout_probs = self.model(x_maskout, self.data.edge_index)
                else:
                    masked_probs = self.model(
                        x_masked, self.data.edge_index, self.data.edge_attr
                    )
                    maskout_probs = self.model(
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
                    masked_probs = self.model(x_masked, masked_edge_index)
                    maskout_probs = self.model(x_maskout, maskout_edge_index)
                else:
                    masked_probs = self.model(
                        x_masked,
                        self.data.edge_index,
                        self.data.edge_attr * edge_mask,
                    )
                    maskout_probs = self.model(
                        x_maskout,
                        self.data.edge_index,
                        self.data.edge_attr * (1 - edge_mask),
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
        data = self.dataset[explained_y_idx]
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
            out = self.model(data=self.data)
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
        if Path(os.path.join(self.save_dir, self.save_name)).is_file():
            (
                explained_y,
                edge_masks,
                node_feat_masks,
                computation_time,
            ) = self.load_mask()
            self.explained_y = explained_y
        else:
            self.explained_y = self._get_explained_y()
            edge_masks, node_feat_masks, computation_time = [], [], []
            for explained_y_idx in self.explained_y:
                edge_mask, node_feat_mask, duration_seconds = eval(
                    "self._compute" + self.task
                )(explained_y_idx)
                edge_masks.append(edge_mask)
                node_feat_masks.append(node_feat_mask)
                computation_time.append(duration_seconds)
            if self.save:
                self.save_mask(edge_masks, node_feat_masks, computation_time)
        return self.explained_y, edge_masks, node_feat_masks, computation_time

    def clean_mask(self, edge_masks, node_feat_masks):
        if edge_masks[0] is not None:
            edge_masks = clean(edge_masks)
        if node_feat_masks[0] is not None:
            node_feat_masks = clean(node_feat_masks)
        return edge_masks, node_feat_masks

    def _transform(self, masks, param):
        """Transform masks according to the given strategy (topk, threshold, sparsity) and level."""
        if param is None:
            return masks
        new_masks = []
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
                    unimportant_indices = (-mask).argsort()[param:]
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
            data = next(iter(dataloader))
            logits = self.model(data)
            pred_labels = logits.argmax(-1)
            true_labels = self.dataset.data.y
            cond1 = pred_labels == true_labels
            cond2 = pred_labels != true_labels
            if self.explained_target is not None:
                cond3 = true_labels == self.explained_target
            else:
                cond3 = np.ones(len(self.dataset), dtype=bool)
            if self.pred_type == "correct":
                list_idx = np.where(cond1 & cond3)[0]
            elif self.pred_type == "wrong":
                list_idx = np.where(cond2 & cond3)[0]
            elif self.pred_type == "mix":
                list_idx = np.where(cond3)[0]
            else:
                raise ValueError("pred_type must be correct, wrong or mix.")
            explained_y = np.random.choice(
                list_idx,
                size=min(self.num_explained_y, len(self.dataset)),
                replace=False,
            )
        else:
            logits = self.model(data=self.data)
            pred_labels = logits.argmax(-1).cpu().numpy()
            true_labels = self.data.y.cpu().numpy()
            cond1 = pred_labels == true_labels
            cond2 = pred_labels != true_labels
            if self.explained_target is not None:
                cond3 = np.where(true_labels == self.explained_target)
            else:
                cond3 = np.ones(self.data.num_nodes, dtype=bool)
            if self.pred_type == "correct":
                list_idx = np.where((cond1) & (cond3))[0]
            elif self.pred_type == "wrong":
                list_idx = np.where(cond2 & cond3)[0]
            elif self.pred_type == "mix":
                list_idx = np.where(cond3)[0]
            else:
                raise ValueError("pred_type must be correct, wrong or mix.")
            explained_y = np.random.choice(
                list_idx, size=self.num_explained_y, replace=False
            )
        return explained_y

    def save_mask(self, edge_masks, node_feat_masks, computation_time):
        save_path = os.path.join(self.save_dir, self.save_name)
        with open(save_path, "wb") as f:
            pickle.dump(
                [self.explained_y, edge_masks, node_feat_masks, computation_time], f
            )

    def load_mask(self):
        save_path = os.path.join(self.save_dir, self.save_name)
        with open(save_path, "rb") as f:
            w_list = pickle.load(f)
        explained_y, edge_masks, node_feat_masks, computation_time = tuple(w_list)
        self.explained_y = explained_y
        return explained_y, edge_masks, node_feat_masks, computation_time
