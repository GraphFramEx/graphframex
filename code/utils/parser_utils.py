import argparse


def get_data_args(data, args):
    if eval(args.explain_graph):
        if args.dataset == "mutag":
            args.num_classes = 2
            args.input_dim = 7
    else:
        args.num_classes = data.num_classes
        args.input_dim = data.x.size(1)
        if args.num_top_edges == -1:
            if args.dataset == "syn1":
                args.num_top_edges = 6
                args.num_basis = 300
            elif args.dataset == "syn2":
                args.num_top_edges = 6
                args.num_basis = 0
            elif args.dataset == "syn3":
                args.num_top_edges = 12
                args.num_basis = 300
            elif args.dataset == "syn4":
                args.num_top_edges = 6
                args.num_basis = 511
            elif args.dataset == "syn5":
                args.num_top_edges = 12
                args.num_basis = 511
            elif args.dataset == "syn6":
                args.num_top_edges = 5
                args.num_basis = 300
    return args


def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dest", type=str, default="/Users/kenzaamara/GithubProjects/Explain")

    parser.add_argument("--seed", help="random seed", type=int, default=10)

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
    parser.add_argument("--num_basis", help="number of nodes in graph", type=int)
    parser.add_argument("--num_shapes", help="number of houses", type=int)

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
        "--num_workers", dest="num_workers", type=int, help="Number of workers to load data.", default=1
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
    parser.add_argument("--num_test", help="number of testing entities (graphs or nodes)", type=int)
    parser.add_argument("--threshold", help="threshold to select edges in mask", type=float, default=-1)
    parser.add_argument("--sparsity", help="ratio of edges to remove from mask", type=float, default=-1)
    parser.add_argument("--topk", help="num top k edges to keep in mask", type=int, default=-1)
    parser.add_argument("--num_top_edges", help="number of edges to keep in explanation", type=int, default=-1)
    parser.add_argument(
        "--true_label", help="do you take target as true label or predicted label", type=str, default="True"
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
        explain_graph="False",
        dataset="syn1",
        num_basis=300,
        num_shapes=150,
        num_test=10,
        opt="adam",  # opt_parser
        opt_scheduler="none",
        max_nodes=100,
        cuda="1",
        feature_type="default",
        lr=0.001,
        clip=2.0,
        batch_size=20,
        num_epochs=1000,
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.1,
        input_dim=1,
        hidden_dim=20,
        output_dim=20,
        num_classes=4,
        num_gc_layers=3,
        dropout=0.0,
        weight_decay=0.005,
        method="base",
        name_suffix="",
        edge_ent=1,
        edge_size=0,
        explainer_name="gnnexplainer",
    )
    return parser.parse_args()
