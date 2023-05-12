import os
from explain import explain_main
import torch
import yaml
import numpy as np
import pandas as pd
import random
from gnn.model import get_gnnNets
from train_gnn import TrainModel
from gendata import get_dataset
from utils.parser_utils import (
    arg_parse,
    create_args_group,
    fix_random_seed,
    get_data_args,
    get_graph_size_args,
)
from pathlib import Path
from torch_geometric.utils import degree


def main(args, args_group):
    fix_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_params = args_group["dataset_params"]
    model_params = args_group["model_params"]

    dataset = get_dataset(
        dataset_root=args.data_save_dir,
        **dataset_params,
    )
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    args = get_data_args(dataset, args)
    model_params["edge_dim"] = args.edge_dim

    data_y = dataset.data.y.cpu().numpy()
    if args.num_classes == 2:
        y_cf_all = 1 - data_y
    else:
        y_cf_all = []
        for y in data_y:
            y_cf_all.append(y + 1 if y < args.num_classes - 1 else 0)
    args.y_cf_all = torch.FloatTensor(y_cf_all).to(device)

    # Statistics of the dataset
    # Number of graphs, number of node features, number of edge features, average number of nodes, average number of edges
    info = {
        "num_graphs": len(dataset),
        "num_nf": args.num_node_features,
        "num_ef": args.edge_dim,
        "avg_num_nodes": np.mean([data.num_nodes for data in dataset]),
        "avg_num_edges": np.mean([data.edge_index.shape[1] for data in dataset]),
        "avg_degree": np.mean(
            [degree(data.edge_index[0]).mean().item() for data in dataset]
        ),
        "num_classes": args.num_classes,
    }
    print(info)

    # Select unseen data to test generalization capacity
    if eval(args.unseen) and eval(args.graph_classification):
        n, num_test = len(dataset), int(len(dataset) * 0.1)
        list_index = random.sample(range(n), num_test)
        unseen_mask = np.zeros(n, dtype=bool)
        unseen_mask[list_index] = True
        unseen_dataset = dataset[unseen_mask]
        unseen_dataset.data, unseen_dataset.slices = dataset.collate(
            [data for data in unseen_dataset]
        )
        dataset = dataset[~unseen_mask]
        dataset.data, dataset.slices = dataset.collate([data for data in dataset])
        args.num_explained_y = int(len(dataset) * 0.1)

    if len(dataset) > 1:
        args.max_num_nodes = max([d.num_nodes for d in dataset])
    else:
        args.max_num_nodes = dataset.data.num_nodes

    if eval(args.graph_classification):
        args.data_split_ratio = [args.train_ratio, args.val_ratio, args.test_ratio]
        dataloader_params = {
            "batch_size": args.batch_size,
            "random_split_flag": eval(args.random_split_flag),
            "data_split_ratio": args.data_split_ratio,
            "seed": args.seed,
        }
    model = get_gnnNets(args.num_node_features, args.num_classes, model_params)
    model_save_name = f"{args.model_name}_{args.num_layers}l_{str(device)}"
    if args.dataset_name.startswith(tuple(["uk", "ieee"])):
        model_save_name = f"{args.datatype}_" + model_save_name
    if eval(args.graph_classification):
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=model_save_name,
            dataloader_params=dataloader_params,
        )
    else:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=model_save_name,
        )
    if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
        trainer.load_model()
    else:
        trainer.train(
            train_params=args_group["train_params"],
            optimizer_params=args_group["optimizer_params"],
        )
    _, _, _, _, _ = trainer.test()

    explain_main(dataset, trainer.model, device, args, unseen=False)
    if eval(args.unseen) and eval(args.graph_classification):
        explain_main(unseen_dataset, trainer.model, device, args, unseen=True)


if __name__ == "__main__":
    parser, args = arg_parse()
    args = get_graph_size_args(args)

    # Get the absolute path to the parent directory of the current file
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Load the config file
    config_path = os.path.join(parent_dir, "configs", "dataset.yaml")
    # read the configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # loop through the config and add any values to the parser as arguments
    for key, value in config[args.dataset_name].items():
        setattr(args, key, value)

    args_group = create_args_group(parser, args)
    main(args, args_group)
