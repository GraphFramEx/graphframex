import os
from evaluate.mask_utils import get_mask_properties
from explain import Explain
import torch
import numpy as np
import pandas as pd
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
    args = get_data_args(dataset.data, args)
    dataset_params["num_classes"] = len(np.unique(dataset.data.y.cpu().numpy()))
    dataset_params["num_node_features"] = dataset.data.x.size(1)
    if eval(args.graph_classification):
        dataloader_params = {
            "batch_size": args.batch_size,
            "random_split_flag": eval(args.random_split_flag),
            "data_split_ratio": [args.train_ratio, args.val_ratio, args.test_ratio],
            "seed": args.seed,
        }

    model = get_gnnNets(
        dataset_params["num_node_features"], dataset_params["num_classes"], model_params
    )

    if eval(args.graph_classification):
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=f"{args.model_name}_{args.num_layers}l",
            dataloader_params=dataloader_params,
        )
    else:
        trainer = TrainModel(
            model=model,
            dataset=dataset,
            device=device,
            graph_classification=eval(args.graph_classification),
            save_dir=os.path.join(args.model_save_dir, args.dataset_name),
            save_name=f"{args.model_name}_{args.num_layers}l",
        )
    if Path(os.path.join(trainer.save_dir, f"{trainer.save_name}_best.pth")).is_file():
        trainer.load_model()
    else:
        trainer.train(
            train_params=args_group["train_params"],
            optimizer_params=args_group["optimizer_params"],
        )
    _, _, _ = trainer.test()

    additional_args = {
        "dataset_name": args.dataset_name,
        "hidden_dim": args.hidden_dim,
        "num_top_edges": args.num_top_edges,
        "num_layers": args.num_layers,
        "num_node_features": args.num_node_features,
        "num_classes": args.num_classes,
        "model_save_dir": args.model_save_dir,
    }
    save_name = "mask_{}_{}_{}_{}_{}_target{}_{}_{}.pkl".format(
        args.dataset_name,
        args.model_name,
        args.explainer_name,
        args.focus,
        args.num_explained_y,
        args.explained_target,
        args.pred_type,
        args.seed,
    )
    explainer = Explain(
        model=trainer.model,
        dataset=dataset,
        device=device,
        graph_classification=eval(args.graph_classification),
        dataset_name=args.dataset_name,
        explainer_params={**args_group["explainer_params"], **additional_args},
        save_dir=os.path.join(
            args.mask_save_dir, args.dataset_name, args.explainer_name
        ),
        save_name=save_name,
    )

    (
        explained_y,
        edge_masks,
        node_feat_masks,
        computation_time,
    ) = explainer.compute_mask()
    edge_masks, node_feat_masks = explainer.clean_mask(edge_masks, node_feat_masks)

    infos = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "explainer": args.explainer_name,
        "focus": args.focus,
        "mask_nature": args.mask_nature,
        "pred_type": args.pred_type,
        "time": float(format(np.mean(computation_time), ".4f")),
    }

    if edge_masks[0] is None:
        raise ValueError("Edge masks are None")
    params_lst = [eval(i) for i in explainer.transf_params.split(",")]
    params_lst.insert(0, None)
    edge_masks_ori = edge_masks.copy()
    for i, param in enumerate(params_lst):
        params_transf = {explainer.mask_transformation: param}
        edge_masks = explainer._transform(edge_masks_ori, param)
        # Compute mask properties
        edge_masks_properties = get_mask_properties(edge_masks)
        # Evaluate scores of the masks
        accuracy_scores, fidelity_scores = explainer.eval(edge_masks, node_feat_masks)
        if accuracy_scores is None:
            scores = {
                key: value
                for key, value in sorted(
                    infos.items()
                    | edge_masks_properties.items()
                    | fidelity_scores.items()
                    | params_transf.items()
                )
            }
        else:
            scores = {
                key: value
                for key, value in sorted(
                    infos.items()
                    | {"top_acc": args.top_acc, "num_top_edges": args.num_top_edges}
                    | edge_masks_properties.items()
                    | accuracy_scores.items()
                    | fidelity_scores.items()
                    | params_transf.items()
                )
            }
        if i == 0:
            results = pd.DataFrame({k: [v] for k, v in scores.items()})
        else:
            results = results.append(scores, ignore_index=True)
    ### Save results ###
    save_path = os.path.join(
        args.result_save_dir, args.dataset_name, args.explainer_name
    )
    os.makedirs(save_path, exist_ok=True)
    results.to_csv(
        os.path.join(
            save_path,
            "results_{}_{}_{}_{}_{}_{}_target{}_{}_{}.csv".format(
                args.dataset_name,
                args.model_name,
                args.explainer_name,
                args.focus,
                args.mask_nature,
                args.num_explained_y,
                args.explained_target,
                args.pred_type,
                args.seed,
            ),
        )
    )


if __name__ == "__main__":
    parser, args = arg_parse()
    args = get_graph_size_args(args)

    if args.dataset_name.startswith(tuple(["ba", "tree"])):
        (
            args.graph_classification,
            args.num_layers,
            args.hidden_dim,
            args.num_epochs,
            args.lr,
            args.weight_decay,
            args.dropout,
            args.readout,
        ) = ("False", 3, 20, 1000, 0.001, 5e-3, 0.0, "identity")

    elif args.dataset_name.startswith(tuple(["cora", "citeseer", "pubmed"])):
        (
            args.graph_classification,
            args.num_layers,
            args.hidden_dim,
            args.num_epochs,
            args.lr,
            args.weight_decay,
            args.dropout,
            args.readout,
        ) = ("False", 2, 16, 200, 0.01, 5e-4, 0.5, "identity")
    elif args.dataset_name == "mutag":
        (
            args.graph_classification,
            args.num_layers,
            args.hidden_dim,
            args.num_epochs,
            args.lr,
            args.weight_decay,
            args.dropout,
            args.readout,
            args.batch_size,
        ) = ("True", 3, 16, 200, 0.001, 5e-4, 0.0, "max", 32)

    args_group = create_args_group(parser, args)
    main(args, args_group)
