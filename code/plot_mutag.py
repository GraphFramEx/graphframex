#%%
%matplotlib inline
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from gendata import get_dataset
from utils.plot_utils import plot_explained_graph
from utils.path import DATA_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR, MASK_DIR


#%%
data_save_dir = DATA_DIR
model_save_dir = MODEL_DIR
mask_save_dir = MASK_DIR

params = {
    "dataset_name": "mutag",
    "graph_classification": "True",
    "num_layers": 3,
    "hidden_dim": 16,
    "num_epochs": 200,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "dropout": 0.0,
    "readout": "max",
    "batch_size": 32,
    "model_name": "gcn",
    "explainer_name": "ig",
    "focus": "phenomenon",
    "num_explained_y": 5,
    "explained_target": 0,
    "pred_type": "correct",
    "num_top_edges": 3,
    "seed": 0,
}

#%%

dataset = get_dataset(
    dataset_root=data_save_dir,
    **params,
)
dataset.data.x = dataset.data.x.float()
dataset.data.y = dataset.data.y.squeeze().long()

#%%
save_dir = os.path.join(mask_save_dir, params["dataset_name"], params["explainer_name"])
save_name = "mask_{}_{}_{}_{}_{}_target{}_{}_{}.pkl".format(
    params["dataset_name"],
    params["model_name"],
    params["explainer_name"],
    params["focus"],
    params["num_explained_y"],
    params["explained_target"],
    params["pred_type"],
    params["seed"],
)

save_path = os.path.join(save_dir, save_name)
with open(save_path, "rb") as f:
    w_list = pickle.load(f)
explained_y, edge_masks, node_feat_masks, computation_time = tuple(w_list)

#%%
gid_list = explained_y
for i, gid in enumerate(gid_list):
    plot_explained_graph(dataset[gid], edge_masks[i], gid, topk=params["num_top_edges"])  # args.num_top_edges

# %%
