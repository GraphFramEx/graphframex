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

# Python 2
try:
    input = raw_input
except NameError:
    pass

basename = os.path.basename

IS_FAIR_CLUSTER = os.path.exists("checkpoint")
IS_AWS_CLUSTER = os.path.exists("checkpoints")
IS_NEW_CLUSTER = IS_FAIR_CLUSTER or IS_AWS_CLUSTER

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
    assert len(dirs) - base_level == 3, "Experiment should be 2 levels below main checkpoint directory"

    dirs.append(datetime.datetime.now(tz=GMT1()).strftime('%Y%m%d_%H%M%S'))
    dst = join(trash_dir, "_".join(dirs[base_level:]))
    print("Moving %s to %s" % (expe, dst))
    shutil.move(expe, dst)


if IS_NEW_CLUSTER:
    blacklist = []
    sbatch = "bash "
    if len(blacklist) >= 1:
        sbatch += "--exclude "
        sbatch += ",".join(blacklist)
        sbatch += " "


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-root", type=str, default=ckpt_default())
parser.add_argument("--trash-dir", type=str, default=join(ckpt_default(), "trash"))
subparsers = parser.add_subparsers(dest='command')

parser_count = subparsers.add_parser("count")
parser_count.add_argument("grid", type=str)

parser_list = subparsers.add_parser("list")
parser_list.add_argument("group", type=str)

parser_remove = subparsers.add_parser("remove")
parser_remove.add_argument("expe", type=str)

parser_sweep = subparsers.add_parser('sweep')
parser_sweep.add_argument('grid', type=str)
parser_sweep.add_argument("--no-launch", action='store_false', dest='launch')
parser_sweep.add_argument("--sample", type=int, default=-1)
parser_sweep.add_argument("--numeric", action='store_true', dest='numeric')
parser_sweep.add_argument("--array", type=bool_flag, default=True)
parser_sweep.add_argument("--pooling", type=int, default=1)
parser_sweep.set_defaults(launch=True, numeric=False)

parser_status = subparsers.add_parser("status")

parser_stress = subparsers.add_parser('stress')
parser_stress.add_argument("--partition", type=str, default="")

parser_nb = subparsers.add_parser('notebook')
parser_nb.add_argument("json_path", type=str)
parser_nb.add_argument("nb_path", type=str)

args = parser.parse_args()

if args.command == 'sweep':
    assert os.path.exists(args.grid), "Config file %s does not exist. Are you in the right repository ?" % args.grid
    config = json.load(open(args.grid))
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

        """
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
                
        """

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

