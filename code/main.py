import os
import random
from code.utils.io_utils import check_dir, create_data_filename, create_model_filename, load_ckpt, save_checkpoint
import torch

from code.dataset.gen_syn import build_syndata

from code.utils.parser_utils import get_data_args

from code.gnn.eval import gnn_scores_nc
from code.gnn.model import GcnEncoderNode

from exp_mutag.gnn_train import train_graph_classification


def main(args):
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate, Save, Load data
    check_dir(args.data_save_dir)
    data_filename = create_data_filename(args)

    if os.path.isfile(data_filename):
        data = torch.load(data_filename)
    else:
        data = build_syndata(args)
        torch.save(data, data_filename)

    data = data.to(device)
    args = get_data_args(data, args)

    # Create, Train, Save, Load GNN model
    model_filename = create_model_filename(args)
    if os.path.isfile(model_filename):
        model = GcnEncoderNode(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args=args
        )

    else:
        model = GcnEncoderNode(
            args.input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers, args=args
        )
        train_graph_classification(model, data, device, args)
        model.eval()
        results_train, results_test = gnn_scores_nc(model, data)
        save_checkpoint(model, args, results_train, results_test)

    ckpt = load_ckpt(model_filename)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    print("__gnn_train_scores: ", ckpt["results_train"])
    print("__gnn_test_scores: ", ckpt["results_test"])

    # Explain
