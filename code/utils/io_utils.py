""" io_utils.py
    Utilities for reading and writing logs.
"""
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

    name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim)
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
    filename = os.path.join(subdir, f"{args.dataset}.pt")
    return filename


def create_model_filename(args, isbest=False, num_epochs=-1):
    """
    Args:
        args        :  the arguments parsed in the parser
        isbest      :  whether the saved model is the best-performing one
        num_epochs  :  epoch number of the model (when isbest=False)
    """
    subdir = os.path.join(args.model_save_dir, args.dataset)
    check_dir(subdir)
    filename = os.path.join(subdir, gen_prefix(args))
    os.makedirs(filename, exist_ok=True)

    if isbest:
        filename = os.path.join(filename, "best")
    filename += f"_gcn_{args.num_gc_layers}"
    return filename + ".pth.tar"


def save_checkpoint(model, args, results_train, results_test, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.
    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    filename = create_model_filename(args, isbest, num_epochs=args.num_epochs)
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


def load_ckpt(filename, isbest=False):
    """Load a pre-trained pytorch model from checkpoint."""
    print("loading model")
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
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


def load_model(path):
    """Load a pytorch model."""
    model = torch.load(path)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    return


def gen_train_plt_name(args):
    return "results/" + gen_prefix(args) + ".png"
