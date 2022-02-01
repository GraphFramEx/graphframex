import argparse



def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dest', type=str, default='/Users/kenzaamara/GithubProjects/Explain')

    parser.add_argument('--seed', help='random seed', type=int, default=41)

    # Computing power
    parser.add_argument("--cuda", dest="cuda", help="CUDA.")
    parser.add_argument(
        "--gpu",
        dest="gpu",
        default=False,
        help="whether to use GPU."
    )


    # saving data
    parser.add_argument('--data_save_dir', help="Directory where benchmark is located", type=str, default='data')
    parser.add_argument('--dataset', type=str)

    # saving model
    parser.add_argument('--model_save_dir', help='saving directory for gnn model', type=str, default='model')

    # build ba-shape graphs
    parser.add_argument('--num_basis', help='number of nodes in graph', type=int)
    parser.add_argument('--num_shapes', help='number of houses', type=int)

    parser.add_argument(
        "--max_nodes",
        dest="max_nodes",
        type=int,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )


    # training parameters
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--bs", type=int)


    parser.add_argument("--batch_size", dest="batch_size", type=int, help="Batch size.")
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
        default=1
    )
    parser.add_argument(
        "--feature_type",
        dest="feature_type",
        help="Feature used for encoder. Can be: id, deg",
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

    parser.add_argument(
        "--method", dest="method", help="Method. Possible values: base, "
    )
    parser.add_argument(
        "--name_suffix", dest="name_suffix", help="suffix added to the output filename"
    )

    # explainer params
    parser.add_argument('--explain_graph', help='graph classification or node classification', type=bool, default=True)
    parser.add_argument('--num_test', help='number of testing entities (graphs or nodes)', type=int)
    parser.add_argument('--threshold', help='threshold to select edges in mask', type=float, default=-1)
    parser.add_argument('--sparsity', help='ratio of edges to remove from mask', type=float, default=-1)
    parser.add_argument('--topk', help='num top k edges to keep in mask', type=int, default=-1)
    parser.add_argument('--num_top_edges', help='number of edges to keep in explanation', type=int, default=-1)
    parser.add_argument('--true_label', help='do you take target as true label or predicted label', type=str,
                        default='True')
    parser.add_argument('--explainer_name', help='explainer', type=str)

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        dataset="syn1",
        num_basis=300,
        num_shapes=150,
        num_test=100,
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
        explainer_name="pagerank"
    )
    return parser.parse_args()