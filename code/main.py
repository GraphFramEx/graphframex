from hashlib import new
import json
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F

from dataset.gen_real import REAL_DATA, WEBKB
from evaluate.accuracy import eval_accuracy
from evaluate.fidelity import eval_fidelity, eval_related_pred_nc
from evaluate.mask_utils import clean_all_masks, get_mask_properties, transform_mask
from explainer.genmask import compute_masks
from gnn.train import get_trained_model
from utils.gen_utils import get_test_nodes, load_data
from utils.io_utils import create_result_filename
from utils.parser_utils import arg_parse, get_data_args


def main(args, data_type):

    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = load_data(args, device, data_type)
    args = get_data_args(data, args)

    # Load model
    model = get_trained_model(data, args, device, data_type="syn")
    model.eval()

    ### Explain ###
    list_test_nodes = get_test_nodes(data, model, args)

    # Compute masks
    node_indices, edge_masks, node_feat_masks, Time = compute_masks(
        list_test_nodes, model, data, device, args
    )
    args.E = False if edge_masks[0] is None else True
    args.NF = False if node_feat_masks[0] is None else True
    if args.NF:
        if node_feat_masks[0].size <= 1:
            args.NF = False
            print("No node feature mask")
    args.num_test_final = len(edge_masks) if args.E else None

    infos = {
        "dataset": args.dataset,
        "model": args.model,
        "explainer": args.explainer_name,
        "n_edges": data.edge_index.size(1),
        "n_expl_nodes": args.num_test,
        "n_expl_nodes_final": args.num_test_final,
        "phenomenon": args.true_label_as_target,
        "hard_mask": args.hard_mask,
        "time": float(format(np.mean(Time), ".4f")),
    }
    print("__infos:" + json.dumps(infos))

    ### Mask normalisation and cleaning ###
    edge_masks, node_feat_masks = clean_all_masks(edge_masks, node_feat_masks, args)

    if eval(args.top_acc):
        ### Accuracy Top ###
        accuracy_top = eval_accuracy(
            data, edge_masks, list_test_nodes, args, top_acc=True
        )
        print("__accuracy_top:" + json.dumps(accuracy_top))

    else:
        ### Transformed mask ###
        params_lst = [eval(i) for i in args.params_list.split(",")]
        params_lst.insert(0, None)
        edge_masks_ori = edge_masks.copy()
        for i, param in enumerate(params_lst):
            if param is None:
                print("Masks are not transformed")
            else:
                print("Masks are transformed with strategy: " + args.strategy)
            params_transf = {args.strategy: param}
            ### Mask transformation ###
            edge_masks = transform_mask(edge_masks_ori, data, param, args)
            # Compute mask properties
            edge_masks_properties = get_mask_properties(edge_masks, data.edge_index)
            edge_masks_properties_transf = {
                key: value
                for key, value in sorted(
                    edge_masks_properties.items() | params_transf.items()
                )
            }
            print("__edge_mask_properties:" + json.dumps(edge_masks_properties_transf))

            if data_type == "syn":
                ### Accuracy ###
                accuracy = eval_accuracy(
                    data, edge_masks, list_test_nodes, args, top_acc=False
                )
                accuracy_scores = {
                    key: value
                    for key, value in sorted(accuracy.items() | params_transf.items())
                }
                print("__accuracy:" + json.dumps(accuracy_scores))

            ### Fidelity ###
            related_preds = eval_related_pred_nc(
                model, data, edge_masks, node_feat_masks, list_test_nodes, device, args
            )
            fidelity = eval_fidelity(related_preds, args)
            fidelity_scores = {
                key: value
                for key, value in sorted(fidelity.items() | params_transf.items())
            }
            print("__fidelity:" + json.dumps(fidelity_scores))

            ### Full results ###
            if data_type == "syn":
                row = {
                    key: value
                    for key, value in sorted(
                        infos.items()
                        | edge_masks_properties.items()
                        | accuracy.items()
                        | fidelity.items()
                        | params_transf.items()
                    )
                }
            else:
                row = {
                    key: value
                    for key, value in sorted(
                        infos.items()
                        | edge_masks_properties.items()
                        | fidelity.items()
                        | params_transf.items()
                    )
                }
            if i == 0:
                results = pd.DataFrame({k: [v] for k, v in row.items()})
            else:
                results = results.append(row, ignore_index=True)
        ### Save results ###
        results.to_csv(create_result_filename(args))


if __name__ == "__main__":
    args = arg_parse()
    if args.dataset.startswith(tuple(["ba", "tree"])):
        (
            args.num_gc_layers,
            args.hidden_dim,
            args.output_dim,
            args.num_epochs,
            args.lr,
            args.weight_decay,
            args.dropout,
        ) = (3, 20, 20, 1000, 0.001, 5e-3, 0.0)
        main(args, data_type="syn")
    elif args.dataset in REAL_DATA.keys():
        if args.dataset in WEBKB.keys():
            (
                args.num_gc_layers,
                args.hidden_dim,
                args.output_dim,
                args.num_epochs,
                args.lr,
                args.weight_decay,
                args.dropout,
            ) = (2, 32, 32, 400, 0.001, 5e-3, 0.2)
        else:
            (
                args.num_gc_layers,
                args.hidden_dim,
                args.output_dim,
                args.num_epochs,
                args.lr,
                args.weight_decay,
                args.dropout,
            ) = (2, 16, 16, 200, 0.01, 5e-4, 0.5)
        main(args, data_type="real")
    elif args.dataset.startswith("ebay"):
        (
            args.num_gc_layers,
            args.hidden_dim,
            args.output_dim,
            args.num_epochs,
            args.lr,
            args.weight_decay,
            args.dropout,
        ) = (2, 32, 32, 500, 0.001, 5e-4, 0.5)
        main(args, data_type="real")
    else:
        pass
