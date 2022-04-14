import argparse
import numpy as np


def get_graph_size_args(args):
    if not eval(args.explain_graph):
        if args.dataset == "syn1":
            args.num_top_edges = 6
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
        elif args.dataset == "syn2":
            args.num_top_edges = 6
            args.num_shapes = 100
            args.width_basis = 350
            args.num_basis = args.width_basis * 2
        elif args.dataset == "syn3":
            args.num_top_edges = 12
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
        elif args.dataset == "syn4":
            args.num_top_edges = 6
            args.num_shapes = 60
            args.width_basis = 8
            args.num_basis = 2 ^ (args.width_basis + 1) - 1
        elif args.dataset == "syn5":
            args.num_top_edges = 12
            args.num_shapes = 80
            args.width_basis = 8
            args.num_basis = 2 ^ (args.width_basis + 1) - 1
        elif args.dataset == "syn6":
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
        if args.dataset.startswith("syn"):
            args.num_classes = data.num_classes
            args.input_dim = data.x.size(1)
        else:
            args.num_classes = len(np.unique(data.y.cpu().numpy()))
            args.input_dim = data.x.size(1)
    return args


def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dest", type=str, default="/Users/kenzaamara/GithubProjects/Explain")

    parser.add_argument("--seed", help="random seed", type=int, default=0)

    # Computing power
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")

    # saving data, model, figures
    parser.add_argument("--data_save_dir", help="Directory where benchmark is located", type=str, default="data")
    parser.add_argument("--model_save_dir", help="saving directory for gnn model", type=str, default="model")
    parser.add_argument("--fig_save_dir", help="Directory where figures are saved", type=str, default="figures")
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

    # sampling
    parser.add_argument(
        "--sample_size",
        dest="sample_size",
        type=int,
        help="Number of nodes in each sample (ClusterSampling).",
    )

    parser.add_argument(
        "--max_nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )

    # training parameters
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--bs", type=int)

    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, help="Number of epochs to train.")
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
        "--num_workers", dest="num_workers", type=int, help="Number of workers to load data.", default=4
    )

    parser.add_argument(
        "--fastmode",
        dest="fastmode",
        type=str,
        help="Evaluate val set performance separately, deactivate dropout during val run",
        default="False",
    )

    parser.add_argument(
        "--feature_type",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
    )
    # gnn achitecture parameters
    parser.add_argument("--input_dim", dest="input_dim", type=int, help="Input feature dimension")
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, help="Hidden dimension")
    parser.add_argument("--output_dim", dest="output_dim", type=int, help="Output dimension")
    parser.add_argument("--num_classes", dest="num_classes", type=int, help="Number of label classes")
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
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--weight_decay",
        dest="weight_decay",
        type=float,
        help="Weight decay regularization constant.",
    )

    parser.add_argument("--method", dest="method", help="Method. Possible values: base, ")
    parser.add_argument("--name_suffix", dest="name_suffix", help="suffix added to the output filename")

    # explainer params
    parser.add_argument("--explain_graph", help="graph classification or node classification", type=str)
    parser.add_argument("--true_label_as_target", help="target is groudtruth label or GNN initial prediction", type=str)
    parser.add_argument("--hard_mask", help="Soft or hard mask", type=str)
    parser.add_argument("--testing_pred", help="True if all testing nodes are correct; False if all testing nodes labels are wrong; None otherwise", type=str, default="mix") # ["correct", "wrong", "mix"]
    
    
    parser.add_argument("--num_test", help="number of testing entities (graphs or nodes)", type=int)
    parser.add_argument("--num_test_final", help="number of testing entities (graphs or nodes) in the final set", type=int)
    parser.add_argument("--time_limit", help="max time for a method to run on testing set", type=int, default=30000)
    
    parser.add_argument("--threshold", help="threshold to select edges in mask", type=float, default=-1)
    parser.add_argument("--sparsity", help="ratio of edges to remove from mask", type=float, default=-1)
    parser.add_argument("--topk", help="num top k edges to keep in mask", type=int, default=-1)
    parser.add_argument("--topk_list", help="list of top k values", type=str, default="")
    parser.add_argument("--num_top_edges", help="number of edges to keep in explanation", type=int, default=-1)
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
        explain_graph="False",
        true_label_as_target="True",
        hard_mask="True",
        dataset="syn1",
        width_basis=300,
        num_shapes=150,
        num_test=10,
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.01,
        clip=2.0,
        sample_size=10e5,
        batch_size=20,
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
        method="base",
        name_suffix="",
        edge_ent=1.0,
        edge_size=0.005,
        explainer_name="gnnexplainer",
    )
    return parser.parse_args()
