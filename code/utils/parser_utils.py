import argparse
import numpy as np
from utils.path import DATA_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR, MASK_DIR


def get_graph_size_args(args):
    if not eval(args.explain_graph):
        if args.dataset == "ba_house":
            args.num_top_edges = 6
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
        elif args.dataset == "ba_community":
            args.num_top_edges = 6
            args.num_shapes = 100
            args.width_basis = 350
            args.num_basis = args.width_basis * 2
        elif args.dataset == "ba_grid":
            args.num_top_edges = 12
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
        elif args.dataset == "tree_cycle":
            args.num_top_edges = 6
            args.num_shapes = 60
            args.width_basis = 8
            args.num_basis = 2 ^ (args.width_basis + 1) - 1
        elif args.dataset == "tree_grid":
            args.num_top_edges = 12
            args.num_shapes = 80
            args.width_basis = 8
            args.num_basis = 2 ^ (args.width_basis + 1) - 1
        elif args.dataset == "ba_bottle":
            args.num_top_edges = 5
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
    return args


def get_data_args(data, args):
    if eval(args.explain_graph):
        if args.dataset == "mutag":
            args.num_classes = 2
            args.input_dim = 7
    else:
        if args.dataset.startswith(tuple(["ba", "tree"])):
            args.num_classes = data.num_classes
            args.input_dim = data.x.size(1)
        else:
            args.num_classes = len(np.unique(data.y.cpu().numpy()))
            args.input_dim = data.x.size(1)
    return args


def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", help="random seed", type=int, default=0)
    parser.add_argument(
        "--dest", help="dest", type=str, default="/cluster/home/kamara/"
    )
    # saving data, model, figures
    parser.add_argument(
        "--save_mask", help="If we save the masks", type=str, default="False"
    )

    parser.add_argument(
        "--data_save_dir",
        help="Directory where benchmark is located",
        type=str,
        default=DATA_DIR,
    )
    parser.add_argument(
        "--logs_save_dir",
        help="Directory where logs are saved",
        type=str,
        default=LOG_DIR,
    )
    parser.add_argument(
        "--model_save_dir",
        help="saving directory for gnn model",
        type=str,
        default=MODEL_DIR,
    )
    parser.add_argument(
        "--mask_save_dir",
        help="Directory where masks are saved",
        type=str,
        default=MASK_DIR,
    )
    parser.add_argument(
        "--result_save_dir",
        help="Directory where results are saved",
        type=str,
        default=RESULT_DIR,
    )
    parser.add_argument(
        "--fig_save_dir",
        help="Directory where figures are saved",
        type=str,
        default="figures",
    )
    parser.add_argument(
        "--draw_graph",
        help="Draw explanations (subgraph for NC and graph for GC) after training",
        type=str,
        default="False",
    )

    # dataset
    parser.add_argument("--dataset", type=str)
    # build ba-shape graphs
    parser.add_argument("--width_basis", help="width of base graph", type=int)
    parser.add_argument("--num_shapes", help="number of houses", type=int)

    # sampling - if dataset is too large, we sample it
    parser.add_argument(
        "--sample_size",
        dest="sample_size",
        type=int,
        help="Number of nodes in each sample (ClusterSampling).",
        default=10e5,
    )

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--weight_decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument(
        "--num_epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser.add_argument(
        "--train_ratio",
        dest="train_ratio",
        type=float,
        help="Ratio of number of graphs testing set to all graphs.",
    )
    parser.add_argument(
        "--test_ratio",
        dest="test_ratio",
        type=float,
        help="Ratio of number of graphs testing set to all graphs.",
    )
    parser.add_argument(
        "--val_ratio",
        dest="val_ratio",
        type=float,
        help="Ratio of number of graphs validation set to all graphs.",
    )

    parser.add_argument(
        "--num_workers",
        dest="num_workers",
        type=int,
        help="Number of workers to load data.",
        default=4,
    )

    # gnn achitecture parameters
    parser.add_argument(
        "--input_dim", dest="input_dim", type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--hidden_dim", dest="hidden_dim", type=int, help="Hidden dimension"
    )
    parser.add_argument(
        "--output_dim", dest="output_dim", type=int, help="Output dimension"
    )
    parser.add_argument(
        "--num_classes", dest="num_classes", type=int, help="Number of label classes"
    )
    parser.add_argument(
        "--num_gc_layers",
        dest="num_gc_layers",
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--bn",
        dest="bn",
        action="store_const",
        const=True,
        default=False,
        help="Whether batch normalization is used",
    )
    parser.add_argument("--dropout", dest="dropout", type=float, help="Dropout rate.")

    parser.add_argument(
        "--model",
        dest="model",
        help="GNN model. Possible values: base, gat, gcn, gine",
        type=str,
    )

    # explainer params
    parser.add_argument(
        "--explain_graph",
        help="graph classification or node classification",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--true_label_as_target",
        help="target is groudtruth label or GNN initial prediction",
        type=str,
    )
    parser.add_argument("--hard_mask", help="Soft or hard mask", type=str)
    parser.add_argument(
        "--testing_pred",
        help="True if all testing nodes are correct; False if all testing nodes labels are wrong; None otherwise",
        type=str,
        default="mix",
    )  # ["correct", "wrong", "mix"]
    parser.add_argument(
        "--top_acc",
        help="Top accuracy for synthetic dataset only",
        type=str,
        default="False",
    )

    parser.add_argument(
        "--num_test", help="number of testing entities (graphs or nodes)", type=int
    )
    parser.add_argument(
        "--num_test_final",
        help="number of testing entities (graphs or nodes) in the final set",
        type=int,
    )
    parser.add_argument(
        "--time_limit",
        help="max time for a method to run on testing set",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--strategy", help="strategy for mask transformation", type=str, default="topk"
    )  # ["topk", "sparsity", "threshold"]
    parser.add_argument(
        "--params_list", help="list of transformation degrees", type=str, default="5,10"
    )
    parser.add_argument(
        "--directed",
        help="if directed, choose the topk directed edges; otherwise topk undirected (no double counting)",
        type=str,
        default="True",
    )
    parser.add_argument(
        "--num_top_edges",
        help="number of edges to keep in explanation",
        type=int,
        default=-1,
    )
    parser.add_argument("--explainer_name", help="explainer", type=str)

    # hyperparameters for GNNExplainer
    parser.add_argument(
        "--edge_size",
        dest="edge_size",
        type=float,
        help="Constraining edge mask size (high `edge_size` => small edge mask)",
    )
    parser.add_argument(
        "--edge_ent",
        dest="edge_ent",
        type=float,
        help="Constraining edge mask entropy: mask is uniform or discriminative",
    )

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        true_label_as_target="True",
        hard_mask="True",
        dataset="ba_house",
        width_basis=300,
        num_shapes=150,
        num_test=5,
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.01,
        clip=2.0,
        sample_size=10e5,
        num_epochs=200,
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.1,
        input_dim=1,
        hidden_dim=16,
        output_dim=16,
        num_classes=4,
        num_gc_layers=2,
        dropout=0.5,
        weight_decay=5e-4,
        model="gcn",
        edge_ent=1.0,
        edge_size=0.005,
        explainer_name="gnnexplainer",
    )
    args, unknown = parser.parse_known_args()
    return args
