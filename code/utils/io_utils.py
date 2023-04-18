""" io_utils.py
    Utilities for reading and writing logs.
"""
from datetime import datetime
import os
import json
import torch


use_cuda = torch.cuda.is_available()


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def gen_prefix(args, **kwargs):
    """Generate label prefix for a graph model."""
    name = args.dataset
    if eval(args.explain_graph):
        name += "_gc"
    else:
        name += "_nc"
    name += (
        "_h"
        + str(args.hidden_dim)
        + "_o"
        + str(args.output_dim)
        + "_"
        + args.model
        + "_"
        + str(args.num_gc_layers)
        + "_epch"
        + str(args.num_epochs)
        + "_lr"
        + str(args.lr)
        + "_wd"
        + str(args.weight_decay)
        + "_drop"
        + str(args.dropout)
    )
    name += "_" + str(args.seed)
    if "suffix" in kwargs:
        name += "_" + kwargs.get("suffix")
    return name


def gen_explainer_prefix(args):
    """Generate label prefix for a graph explainer model."""
    name = gen_prefix(args) + "_explain"
    if len(args.explainer_suffix) > 0:
        name += "_" + args.explainer_suffix
    return name


def create_data_filename(args):
    subdir = os.path.join(args.data_save_dir, args.dataset)
    os.makedirs(subdir, exist_ok=True)
    filename = os.path.join(
        subdir, f"{args.dataset}_{args.num_shapes}_{args.width_basis}.pt"
    )
    return filename


def create_model_filename(args, isbest=False, num_epochs=-1, **kwargs):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    subdir = os.path.join(args.model_save_dir, args.dataset)
    os.makedirs(subdir, exist_ok=True)
    filename = os.path.join(subdir, gen_prefix(args, **kwargs))

    if isbest:
        filename = os.path.join(filename, "best")
    return filename + ".pth.tar"


def create_mask_filename(args):
    os.makedirs(args.mask_save_dir, exist_ok=True)
    subdir = os.path.join(args.mask_save_dir, args.dataset, args.explainer_name)
    os.makedirs(subdir, exist_ok=True)

    name = args.dataset + "_" + args.explainer_name
    name += (
        "_model_"
        + str(args.model)
        + "_phenfocus_"
        + str(args.true_label_as_target)
        + "_test_"
        + str(args.num_test)
    )
    name += "_seed" + str(args.seed)

    filename = os.path.join(subdir, name)
    return filename + ".pkl"


def create_result_filename(args):
    os.makedirs(args.result_save_dir, exist_ok=True)
    subdir = os.path.join(args.result_save_dir, args.dataset, args.explainer_name)
    os.makedirs(subdir, exist_ok=True)

    name = args.dataset + "_" + args.explainer_name
    name += (
        "_model_"
        + str(args.model)
        + "_phenfocus_"
        + str(args.true_label_as_target)
        + "_hardmask_"
        + str(args.hard_mask)
        + "_test_"
        + str(args.num_test)
    )
    name += "_seed" + str(args.seed)

    filename = os.path.join(subdir, name)
    return filename + ".csv"


def save_checkpoint(
    filename, model, args, results_train, results_test, isbest=False, cg_dict=None
):
    """Save pytorch model checkpoint.
    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    torch.save(
        {
            "model_type": "gcn",
            "epoch": args.num_epochs,
            "model_type": args.explainer_name,
            "optimizer": args.optimizer,
            "results_train": results_train,
            "results_test": results_test,
            "model_state": model.state_dict(),
            "cg": cg_dict,
        },
        filename,
    )


def load_ckpt(filename, device, isbest=False):
    """Load a pre-trained pytorch model from checkpoint."""
    print("loading model")
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename, map_location=device)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def gen_train_plt_name(args):
    save_fig_dir = os.path.join(
        os.path.join(args.model_save_dir, args.dataset), "results"
    )
    os.makedirs(save_fig_dir, exist_ok=True)
    return os.path.join(save_fig_dir, gen_prefix(args)) + ".png"


def gen_mask_density_plt_name(args, type):
    save_fig_dir = os.path.join(args.fig_save_dir, "density")
    save_fig_dir = os.path.join(os.path.join(save_fig_dir, type), args.dataset)
    os.makedirs(save_fig_dir, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        os.path.join(save_fig_dir, args.explainer_name)
        + f"_focus_{args.true_label_as_target}_hard_{args.hard_mask}_sparsity_{args.sparsity}_{date}.png"
    )


def gen_feat_importance_plt_name(args):
    save_fig_dir = os.path.join(args.fig_save_dir, "feat_importance")
    save_fig_dir = os.path.join(save_fig_dir, args.dataset)
    os.makedirs(save_fig_dir, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (
        os.path.join(save_fig_dir, args.explainer_name)
        + f"_focus_{args.true_label_as_target}_hard_{args.hard_mask}_sparsity_{args.sparsity}_{date}.png"
    )


def write_to_json(entry, fname):
    a = []
    if (not os.path.exists(fname)) or (os.stat(fname).st_size == 0):
        a.append(entry)
        with open(fname, mode='w', encoding='utf-8', errors='ignore') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as feedsjson:
            feeds = json.load(feedsjson)
        feeds.append(entry)
        with open(fname, mode='w', encoding='utf-8', errors='ignore') as f:
            f.write(json.dumps(feeds, indent=2))