""" io_utils.py
    Utilities for reading and writing logs.
"""
from datetime import datetime
import os

import torch

# Only necessary to rebuild the Chemistry example
# from rdkit import Chem

use_cuda = torch.cuda.is_available()


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def gen_prefix(args):
    """Generate label prefix for a graph model."""
    name = args.dataset
    if eval(args.explain_graph):
        name += "_gc"
    else:
        name += "_nc"
    name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim) + "_epch" + str(args.num_epochs)
    if not args.bias:
        name += "_nobias"
    if len(args.name_suffix) > 0:
        name += "_" + args.name_suffix
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
    filename = os.path.join(subdir, f"{args.dataset}_{args.num_shapes}_{args.width_basis}.pt")
    return filename


def create_model_filename(args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    subdir = os.path.join(args.model_save_dir, args.dataset)
    os.makedirs(subdir, exist_ok=True)
    filename = os.path.join(subdir, gen_prefix(args))

    if isbest:
        filename = os.path.join(filename, "best")
    filename += f"_gcn_{args.num_gc_layers}"
    return filename + ".pth.tar"


def save_checkpoint(filename, model, args, results_train, results_test, isbest=False, cg_dict=None):
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
    save_fig_dir = os.path.join(os.path.join(args.model_save_dir, args.dataset), "results")
    os.makedirs(save_fig_dir, exist_ok=True)
    return os.path.join(save_fig_dir, gen_prefix(args)) + ".png"


def gen_mask_density_plt_name(args):
    save_fig_dir = os.path.join(args.fig_save_dir, args.dataset)
    os.makedirs(save_fig_dir, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(save_fig_dir, args.explainer_name) + f"_ent_{args.edge_ent}_size_{args.edge_size}_{date}.png"
