import os

import networkx as nx
import numpy as np
from dataset.syn_utils.synthetic_structsim import bottle, cycle, grid, house
from explainer.node_explainer import node_attr_to_edge


def get_ground_truth(node, data, dataset_name):
    gt = []
    if dataset_name == "ba_house":
        gt = get_ground_truth_ba_house(node)  # correct
        graph, role = house(gt[0], role_start=1)
    elif dataset_name == "ba_community":
        gt = get_ground_truth_ba_house(node)  # correct
        role = data.y[gt]
    elif dataset_name == "ba_grid":
        gt = get_ground_truth_ba_grid(node)  # correct
        graph, role = grid(gt[0], dim=3, role_start=1)
    elif dataset_name == "tree_cycle":
        gt = get_ground_truth_tree_cycle(node)  # correct
        graph, role = cycle(gt[0], 6, role_start=1)
    elif dataset_name == "tree_grid":
        gt = get_ground_truth_tree_grid(node)  # correct
        graph, role = grid(gt[0], dim=3, role_start=1)
    elif dataset_name == "ba_bottle":
        gt = get_ground_truth_ba_house(node)  # correct
        graph, role = bottle(gt[0], role_start=1)

    true_node_mask = np.zeros(data.x.shape[0])
    true_node_mask[gt] = 1
    true_edge_mask = node_attr_to_edge(data.edge_index, true_node_mask)
    true_edge_mask = np.where(true_edge_mask == 2, 1, 0)

    if dataset_name == "ba_community":
        graph = nx.Graph()
        graph.add_nodes_from(gt)
        new_edges = np.array(data.edge_index[:, np.where(true_edge_mask > 0)[0]].T)
        graph.add_edges_from(new_edges)

    return graph, role, true_edge_mask


def get_ground_truth_ba_house(node):
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    return ground_truth


def get_ground_truth_ba_grid(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]
    return ground_truth


def get_ground_truth_tree_cycle(node):
    buff = node - 1
    base = [0, 1, 2, 3, 4, 5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]
    return ground_truth


def get_ground_truth_tree_grid(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]
    return ground_truth
