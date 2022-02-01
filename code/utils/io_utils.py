""" io_utils.py
    Utilities for reading and writing logs.
"""
import os
import statistics
import re
import csv

import numpy as np
import pandas as pd
import scipy as sc


import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import networkx as nx
import tensorboardX

import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable

# Only necessary to rebuild the Chemistry example
# from rdkit import Chem

import utils.featgen as featgen

use_cuda = torch.cuda.is_available()



def gen_prefix(args):
    '''Generate label prefix for a graph model.
    '''
    name = args.dataset
    name += "_" + args.method

    name += "_h" + str(args.hidden_dim) + "_o" + str(args.output_dim)
    if not args.bias:
        name += "_nobias"
    if len(args.name_suffix) > 0:
        name += "_" + args.name_suffix
    return name



def gen_train_plt_name(args):
    return "results/" + gen_prefix(args) + ".png"


def check_dir(save_dirs):
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)