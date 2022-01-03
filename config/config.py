import argparse
import json
import sys, os
sys.path.append(os.getcwd())
from math import ceil
from os.path import join
import argparse
import datetime
import json
import numpy as np
import os
import shutil
import subprocess
import tempfile
import time

from date import GMT1
from functions import cmdPre, loadlist, savelist
from functions import run_command, getuser, replaceMacros, bool_flag, linearize_params
from params import enumerateParams, generateExt


basename = os.path.basename

IS_FAIR_CLUSTER = os.path.exists("checkpoint")
IS_AWS_CLUSTER = os.path.exists("checkpoints")
IS_NEW_CLUSTER = IS_FAIR_CLUSTER or IS_AWS_CLUSTER

if IS_NEW_CLUSTER:
    blacklist = []
    sbatch = "sbatch "
    if len(blacklist) >= 1:
        sbatch += "--exclude "
        sbatch += ",".join(blacklist)
        sbatch += " "

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', help='Configuration JSON file', default='exp1_ba_shapes/config_file.json')
    # saving data
    parser.add_argument('--data_save_dir', help='File list by write RTL command', default='data')
    parser.add_argument('--data_name', default='ba_shapes')

    # build ba-shape graphs
    parser.add_argument('--n_basis', help='number of nodes in graph', default=2000)
    parser.add_argument('--n_shapes', help='number of houses', default=200)

    # gnn achitecture parameters
    parser.add_argument('--num_layers', help='number of GCN layers', default=2)
    parser.add_argument('--hidden_dim', help='number of neurons in hidden layers', default=16)

    # training parameters
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=200)

    # saving model
    parser.add_argument('--model_save_dir', help='saving directory for gnn model', default='model')

    # explainer params
    parser.add_argument('--true_label', help='do you take target as true label or predicted label', default=True)
    parser.add_argument('--explainer_name', help='explainer', default='random')


    # Parse argument defined so far (in order to process JSON load
    # All arguments not yet defined and passed in command -line are stored in left_argv
    # and processed AFTER JSON load file
    args, unknown = parser.parse_known_args()


    if args.config_file is not None:
        if '.json' in args.config_file:
            # The escaping of "\t" in the config file is necesarry as
            # otherwise Python will try to treat is as the string escape
            # sequence for ASCII Horizontal Tab when it encounters it
            # during json.load
            config = json.load(open(args.config_file))
            parser.set_defaults(**config)

            [
                parser.add_argument(arg)
                for arg in [arg for arg in unknown if arg.startswith('--')]
                if arg.split('--')[-1] in config
            ]

        args = parser.parse_args()
        return args

def ckpt_default():
    return os.getcwd()


def trash_expe(expe, trash_dir):
    """
    Moves experiment folder to trash_dir
    """
    dirs = [x for x in expe.split("/") if x != ""]
    base_level = len([x for x in ckpt_default().split("/") if x != ""])

    assert os.path.exists(expe), "Experiment does not exist"
    assert expe.startswith(ckpt_default()), "Directory should start with default checkpoint"
    assert len(dirs) - base_level == 2, "Experiment should be 2 levels below main checkpoint directory"

    dirs.append(datetime.datetime.now(tz=GMT1()).strftime('%Y%m%d_%H%M%S'))
    dst = join(trash_dir, "_".join(dirs[base_level:]))
    print("Moving %s to %s" % (expe, dst))
    shutil.move(expe, dst)


def run_jobs(args):

    config = json.load(open(args.config_file))
    if "pwd" not in config:
        config["pwd"] = "."#os.getcwd()

    if "meta" in config:
        group = config["meta"]["group"]
        name = config["meta"]["name"]
        args.dest_arg = (config["meta"]["dest-arg"] == "yes")
        print(config)
    else:
        group = args.group
        name = args.name
    ckpt_root = ckpt_default()
    group_root = join(ckpt_root, group)

    expdir = join(group_root, name)
    log_dir = join(expdir, "logs")
    if os.path.exists(log_dir):
        print(f"Experiment {group_root}/{name} already exists. You can: ")
        print(
            f"- Run it anyway. Any output of a previous experiment with the same hyperparameters will be overwritten. The code used will be the existing code and not a fresh clone.")
        print(
            f"- Trash existing experiment folder and create a fresh one. The (old) experiment folder will be put in the trash ({args.trash_dir})")
        print(f"- (default) Abort. Experiment folder will be kpet untouched and new experiments will not be run.")
        print(f"What do you choose ? [run|trash|abort]")
        answer = input().lower()
        if answer == "run":
            continuing = True
        elif answer == "trash":
            trash_expe(expdir, args.trash_dir)
            os.makedirs(log_dir)
        else:
            import sys;
            sys.exit(0)
    else:
        os.makedirs(log_dir)

    # List config of parameters
    paramset = enumerateParams(config["params"])
    param_names = list(set([k for d in paramset for k in d.keys()]))
    param_values = {k: sorted(list(set([d[k] for d in paramset if k in d]))) for k in param_names}

    params_to_index = [k for k, values in param_values.items() if len(values) >= 2 and (
                any(["/" in v for v in values if type(v) is str]) or any([len(str(v)) >= 20 for v in values]))]

    for i_param, params in enumerate(paramset):
        ext = generateExt(params, param_values, to_index=params_to_index)
        params = replaceMacros(params)

        log_stdout = join(log_dir, ext + ".stdout")
        log_stderr = join(log_dir, ext + ".stderr")

        if args.dest_arg:
            if os.path.exists(join(expdir, ext)):
                print(
                    'WARNING: %s already exists (whatever is in there will probably be overwritten by experiment)' % join(
                        expdir, ext))
            else:
                os.makedirs(join(expdir, ext))
            dest_name = config["meta"]["dest-name"] if "dest-name" in config["meta"] else "dest"
            dest_name = [dest_name] if type(dest_name) is str else dest_name
            for dname in dest_name:
                params[dname] = join(expdir, ext)

        filename = join(expdir, 'run%s.sh' % ext)
        with open(filename, 'w') as f:
            f.write(cmdPre(config, params, name + ext, log_stdout, log_stderr, filename))

        subprocess.check_output(["chmod -R a+w %s" % expdir], shell=True)
        subprocess.check_output(["chmod +x %s" % filename], shell=True)

        # print([sbatch + filename])
        start = time.time()
        r = subprocess.check_output([sbatch + filename], shell=True)
        # jobid = int(r.rstrip().split(" ")[3])
        # print(r)
        print("Took %.2f" % (time.time() - start))
        print(log_stdout)
    return


if __name__ == '__main__':
    args = get_params()
    run_jobs(args)