import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


def split_data(data, args):
    """generates train, val, test splits

    Args:
        data: data object
        args: arguments from command line

    Returns:
        data: data object with train, val, test splits
    """
    n = data.num_nodes
    data.train_mask, data.val_mask, data.test_mask = (
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
        torch.zeros(n, dtype=torch.bool),
    )
    train_ids, test_ids = train_test_split(range(n), test_size=args.test_ratio, random_state=args.seed, shuffle=True)
    train_ids, val_ids = train_test_split(train_ids, test_size=args.val_ratio, random_state=args.seed, shuffle=True)

    data.train_mask[train_ids] = 1
    data.val_mask[val_ids] = 1
    data.test_mask[test_ids] = 1
    return data

def get_split(data, args):
    """extract train, val, test splits from a data object in the list"""
    k = args.seed%10
    data.train_mask = data.train_mask[:,k]
    data.val_mask = data.val_mask[:,k]
    data.test_mask = data.test_mask[:,k]
    return data

