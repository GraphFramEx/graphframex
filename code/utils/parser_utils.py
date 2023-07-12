import argparse
import numpy as np
import torch
import random
from utils.path import DATA_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR, MASK_DIR


def fix_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_graph_size_args(args):
    if not eval(args.graph_classification):
        if args.dataset_name == "ba_house":
            args.num_top_edges = 6
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
        elif args.dataset_name == "ba_community":
            args.num_top_edges = 6
            args.num_shapes = 100
            args.width_basis = 350
            args.num_basis = args.width_basis * 2
        elif args.dataset_name == "ba_grid":
            args.num_top_edges = 12
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
        elif args.dataset_name == "tree_cycle":
            args.num_top_edges = 6
            args.num_shapes = 60
            args.width_basis = 8
            args.num_basis = 2 ^ (args.width_basis + 1) - 1
        elif args.dataset_name == "tree_grid":
            args.num_top_edges = 12
            args.num_shapes = 80
            args.width_basis = 8
            args.num_basis = 2 ^ (args.width_basis + 1) - 1
        elif args.dataset_name == "ba_bottle":
            args.num_top_edges = 5
            args.num_shapes = 80
            args.width_basis = 300
            args.num_basis = args.width_basis
    return args


def get_data_args(dataset, args):
    assert dataset.data.y.ndim == 1  # make sure it is a one class problem
    args.num_classes = max(np.unique(dataset.data.y.cpu().numpy())) + 1
    args.num_node_features = dataset.data.x.size(1)

    if dataset.data.edge_attr.ndim == 1:
        dataset.data.edge_attr = torch.unsqueeze(dataset.data.edge_attr, 1)

    args.edge_dim = dataset.data.edge_attr.size(1)
    args.datatype = "binary" if args.num_classes == 2 else "multiclass"
    return args


def arg_parse():

    parser = argparse.ArgumentParser()

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
    # generalization capacity - test on unseen graphs
    parser.add_argument(
        "--unseen",
        help="Split the dataset into data and unseen data and test on unseen data",
        type=str,
        default="False",
    )

    # dataset parameters
    parser_dataset_params = parser.add_argument_group("dataset_params")
    parser_dataset_params.add_argument("--dataset_name", type=str)
    parser_dataset_params.add_argument(
        "--seed", help="random seed", type=int, default=0
    )
    parser_dataset_params.add_argument(
        "--width_basis", help="width of base graph", type=int
    )
    parser_dataset_params.add_argument(
        "--num_shapes", help="number of houses", type=int
    )
    parser_dataset_params.add_argument(
        "--num_basis", help="number of nodes in the base graph", type=int
    )
    parser_dataset_params.add_argument(
        "--datatype",
        help="the type of classification (binary or multiclass) for the powergrid datasets (uk, ieee24, ieee39, ieee118)",
        type=str,
        default="binary",
    )
    parser_dataset_params.add_argument("--num_classes", help="output_dim", type=int)
    parser_dataset_params.add_argument(
        "--num_node_features", help="input_dim", type=int
    )
    parser.add_argument(
        "--num_top_edges",
        help="# edges in groundtruth explanation (for syn data) or max size for subgraphX (for real data)",
        type=int,
        default=10,
    )
    parser_dataset_params.add_argument("--train_ratio", dest="train_ratio", type=float)
    parser_dataset_params.add_argument("--test_ratio", dest="test_ratio", type=float)
    parser_dataset_params.add_argument("--val_ratio", dest="val_ratio", type=float)
    parser_dataset_params.add_argument("--random_split_flag", type=str, default="True")

    # optimization parameters
    parser_optimizer_params = parser.add_argument_group("optimizer_params")
    parser_optimizer_params.add_argument("--lr", type=float)
    parser_optimizer_params.add_argument(
        "--weight_decay",
        type=float,
    )

    # training parameters
    parser_train_params = parser.add_argument_group("train_params")
    parser_train_params.add_argument(
        "--num_epochs", dest="num_epochs", type=int, help="Number of epochs to train."
    )
    parser_train_params.add_argument(
        "--num_early_stop", type=int, help="Num steps before stopping", default=0
    )
    parser_train_params.add_argument(
        "--milestones", type=int, help="Learning decay step size.", default=None
    )
    parser_train_params.add_argument(
        "--gamma", type=float, help="Learning rate decay.", default=None
    )

    # model parameters
    parser_model_params = parser.add_argument_group("model_params")
    parser_model_params.add_argument(
        "--model_name",
        help="[gat, gcn, gin, transformer]. GCN can only be used for data with no or 1D edge features.",
        type=str,
    )
    parser_model_params.add_argument(
        "--graph_classification",
        help="graph or node classification",
        type=str,
        default="False",
    )
    parser_model_params.add_argument("--hidden_dim", type=int, help="Hidden dimension")

    parser_model_params.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers before pooling",
    )
    parser_model_params.add_argument("--dropout", type=float, help="Dropout rate.")
    parser_model_params.add_argument(
        "--readout", type=str, help="Readout type [mean, sum, max]."
    )
    parser_model_params.add_argument(
        "--edge_dim",
        type=int,
        help="Edge feature dimension (only for GAT, GIN and TRANSFORMER model).",
    )

    # explainer parameters
    parser_explainer_params = parser.add_argument_group("explainer_params")
    parser_explainer_params.add_argument("--explainer_name", help="explainer", type=str)
    parser_explainer_params.add_argument(
        "--groundtruth",
        type=str,
        default="False",
    )
    parser_explainer_params.add_argument(
        "--mu",
        type=float,
        default=0.2,
    )
    parser_explainer_params.add_argument(
        "--focus",
        help="target is groudtruth label or GNN initial prediction",
        type=str,
    )
    parser_explainer_params.add_argument(
        "--mask_nature", help="Soft or hard mask", type=str
    )
    parser_explainer_params.add_argument(
        "--pred_type",
        help="True if all testing nodes are correct; False if all testing nodes labels are wrong; None otherwise",
        type=str,
        default="mix",
    )  # ["correct", "wrong", "mix"]
    parser_explainer_params.add_argument(
        "--explained_target",
        help="the class you only want to explain; None otherwise",
        type=int,
        default=None,
    )
    parser_explainer_params.add_argument(
        "--num_explained_y",
        help="number of explained entities (graphs or nodes)",
        type=int,
    )
    parser_explainer_params.add_argument(
        "--time_limit",
        help="max time for a method to run on testing set",
        type=int,
        default=30000,
    )

    parser_explainer_params.add_argument(
        "--mask_transformation",
        help="strategy for mask transformation",
        type=str,
        default="topk",
    )  # ["topk", "sparsity", "threshold"]
    parser_explainer_params.add_argument(
        "--transf_params",
        help="list of transformation degrees",
        type=str,
        default="[10]",
    )
    parser_explainer_params.add_argument(
        "--directed",
        help="if directed, choose the topk directed edges; otherwise topk undirected (no double counting)",
        type=str,
        default="True",
    )

    parser_explainer_params.add_argument(
        "--num_rounds",
        help="num of perturbed edges in the explanation",
        type=int,
        default=0,
    )

    # hyperparameters for GNNExplainer
    parser_explainer_params.add_argument(
        "--edge_size",
        dest="edge_size",
        type=float,
        help="Constraining edge mask size (high `edge_size` => small edge mask)",
    )
    parser_explainer_params.add_argument(
        "--edge_ent",
        dest="edge_ent",
        type=float,
        help="Constraining edge mask entropy: mask is uniform or discriminative",
    )

    parser.set_defaults(
        datadir="data",  # io_parser
        logdir="log",
        ckptdir="ckpt",
        focus="phenomenon",
        mask_nature="hard",
        dataset_name="ba_house",
        width_basis=300,
        num_shapes=150,
        num_explained_y=5,
        opt="adam",
        lr=0.005,
        num_epochs=400,
        train_ratio=0.8,
        val_ratio=0.15,
        test_ratio=0.1,
        num_node_features=1,
        hidden_dim=20,
        num_classes=4,
        num_layers=3,
        dropout=0,
        readout="identity",
        weight_decay=5e-4,
        model_name="gcn",
        edge_ent=1.0,
        edge_size=0.005,
        explainer_name="gnnexplainer",
    )
    args, unknown = parser.parse_known_args()
    return parser, args


def create_args_group(parser, args):
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = group_dict
    return arg_groups
