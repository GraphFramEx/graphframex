import torch
from torch.utils.data import random_split, Subset
from torch_geometric.data import DataLoader
from dataset import (
    MoleculeDataset,
    SynGraphDataset,
    NCRealGraphDataset,
    IEEE24,
    IEEE39,
    UK,
    IEEE24Cont,
    IEEE39Cont,
    UKCont,
)
from torch import default_generator
from utils.parser_utils import arg_parse, get_graph_size_args


def get_dataset(dataset_root, **kwargs):
    dataset_name = kwargs.get("dataset_name")
    print(f"Loading {dataset_name} dataset...")
    if dataset_name.lower() in list(MoleculeDataset.names.keys()):
        return MoleculeDataset(root=dataset_root, name=dataset_name)
    elif dataset_name.lower() in list(NCRealGraphDataset.names.keys()):
        dataset = NCRealGraphDataset(
            root=dataset_root, name=dataset_name, dataset_params=kwargs
        )
        dataset.process()
        return dataset
    elif dataset_name.lower() in list(SynGraphDataset.names.keys()):
        dataset = SynGraphDataset(root=dataset_root, name=dataset_name, **kwargs)
        dataset.process()
        return dataset
    elif dataset_name.lower().startswith(tuple(["uk", "ieee24", "ieee39"])):
        datatype = "multiclass" if dataset_name.lower().endswith("mc") else "binary"
        if dataset_name.lower() in ["uk_mc", "uk_bin"]:
            return UK(root=dataset_root, name=dataset_name, datatype=datatype)
        elif dataset_name.lower() in ["ieee24_mc", "ieee24_bin"]:
            return IEEE24(root=dataset_root, name=dataset_name, datatype=datatype)
        elif dataset_name.lower() in ["ieee39_mc", "ieee39_bin"]:
            return IEEE39(root=dataset_root, name=dataset_name, datatype=datatype)
        elif dataset_name.lower() in ["ukcont_mc", "ukcont_bin"]:
            return UKCont(root=dataset_root, name=dataset_name, datatype=datatype)
        elif dataset_name.lower() in ["ieee24cont_mc", "ieee24cont_bin"]:
            return IEEE24Cont(root=dataset_root, name=dataset_name, datatype=datatype)
        elif dataset_name.lower() in ["ieee39cont_mc", "ieee39cont_bin"]:
            return IEEE39Cont(root=dataset_root, name=dataset_name, datatype=datatype)
        else:
            raise ValueError(f"{dataset_name} is not defined.")
    else:
        raise ValueError(f"{dataset_name} is not defined.")


def get_dataloader(
    dataset, batch_size, random_split_flag=True, data_split_ratio=None, seed=2
):
    """
    Args:
        dataset:
        batch_size: int
        random_split_flag: bool
        data_split_ratio: list, training, validation and testing ratio
        seed: random seed to split the dataset randomly
    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    if not random_split_flag and hasattr(dataset, "supplement"):
        assert "split_indices" in dataset.supplement.keys(), "split idx"
        split_indices = dataset.supplement["split_indices"]
        train_indices = torch.where(split_indices == 0)[0].numpy().tolist()
        dev_indices = torch.where(split_indices == 1)[0].numpy().tolist()
        test_indices = torch.where(split_indices == 2)[0].numpy().tolist()

        train = Subset(dataset, train_indices)
        eval = Subset(dataset, dev_indices)
        test = Subset(dataset, test_indices)
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        from functools import partial

        train, eval, test = random_split(
            dataset,
            lengths=[num_train, num_eval, num_test],
            generator=default_generator.manual_seed(seed),
        )

    dataloader = dict()
    dataloader["train"] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader["eval"] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    dataloader["test"] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader


if __name__ == "__main__":
    args = arg_parse()
    args = get_graph_size_args(args)
    data_params = {
        "num_shapes": args.num_shapes,
        "width_basis": args.width_basis,
        "input_dim": args.input_dim,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
    }
    dataset = get_dataset(args.data_save_dir, "ba_house", **data_params)
    # dataset = get_dataset(args.data_save_dir, "cora")
    print(dataset)
    print(dataset.data)
