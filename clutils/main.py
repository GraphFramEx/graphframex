import argparse
import json
import sys, os
sys.path.append(os.getcwd())

from collections import OrderedDict
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

from src.date import GMT1
from src.functions import cmdPre, loadlist, savelist
from src.functions import run_command, getuser, replaceMacros, bool_flag, linearize_params
from src.params import enumerateParams, generateExt

basename = os.path.basename

if not os.path.exists("checkpoints"):
    os.makedirs("/cluster/home/kamara/Explain/checkpoints")

def ckpt_default():
    return os.getcwd()

def jobname_default():
    return "/cluster/home/kamara/Explain/jobnames.txt"
    
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
    
def stool_stress(partition):
    """
    Number of used CPU and GPU per user at `partition`.
    """
    STRING = ("bqueues " + partition + " --format \"%t %u %b\" "
              "| grep \"^R\" "
              "| grep -v \"null\" "
              "| cut -b3- "
              "| sed -e \"s/gpu://g\" "
              "| sed -e \"s/volta://g\" "
              "| awk '{arr[$1]+=$2} END {for (i in arr) {print i,arr[i]}}' "
              "| sort -n -k2 "
              "| awk '{printf \"%5s %s\\n\", $2, $1}'")
    subprocess.call(STRING, shell=True)

    STRING = ("bqueues " + partition + " --format \"%t %b\" "
              "| grep gpu "
              "| grep R "
              "| cut -d\":\" -f2 "
              "| paste -sd+ "
              "| bc"
              "| awk '{printf \"%s%s\\n\", \"==== Total GPU: \", $1}'")
    subprocess.call(STRING, shell=True)

    print()

    STRING = ("bqueues " + partition + " --format \"%t %u %C\" "
              "| grep \"^R\" "
              "| grep -v \"null\" "
              "| cut -b3- "
              "| sed -e \"s/gpu://g\" "
              "| sed -e \"s/volta://g\" "
              "| awk '{arr[$1]+=$2} END {for (i in arr) {print i,arr[i]}}' "
              "| sort -n -k2 "
              "| awk '{printf \"%5s %s\\n\", $2, $1}'")
    subprocess.call(STRING, shell=True)

    STRING = ("bqueues " + partition + " --format \"%t %C\" "
              "| grep \"^R\" "
              "| cut -b3- "
              "| paste -sd+ "
              "| bc"
              "| awk '{printf \"%s%s\\n\", \"==== Total CPU: \", $1}'")
    subprocess.call(STRING, shell=True)

    print()

    STRING = ("echo You have "
              "$(squeue -u $(whoami) --format %t | grep R | wc -l) "
              "running and "
              "$(squeue -u $(whoami) --format %t | grep PD | wc -l) "
              "pending jobs.")

    subprocess.call(STRING, shell=True)

    return None


sbatch = "bsub "


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-root", type=str, default=ckpt_default())
parser.add_argument("--jobnames", type=str, default=jobname_default())
parser.add_argument("--trash-dir", type=str, default=join(ckpt_default(), "trash"))
parser.add_argument("--launch-parallel", action='store_true', dest='launch_parallel')
parser.set_defaults(launch_parallel=False)

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
parser_sweep.add_argument("--array", type=bool_flag, default=False)
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
    
    
    if "machine" in config:
        #Task is a node
        if "cpus" in config["machine"]:
            sbatch += "-n %s " % config["machine"]["cpus"]
        
        elif "options" in config["machine"]:
            sbatch += "-R "
            
            if "cpus-per-task" in config["machine"]["options"]:
                sbatch += "'span[ptile=%s]' " % config["machine"]["cpus-per-task"]
            
            if "mem" in config["machine"]:
                sbatch += "'rusage[mem=%s]' " % config["machine"]["mem"]
            
            elif "mem-per-cpu" in config["machine"]["options"]:
                assert "cpus-per-task" in config["machine"]
                mem_cpu = int(config["machine"]["mem-per-cpu"].replace("G", ""))
                cpus = int(config["machine"]["cpus-per-task"])
                sbatch += "'[rusage[mem=%d]]' " % (mem_cpu * cpus)
            
            if "gpu-model" in config["machine"]["options"]:
                sbatch += "'select[gpu_model0=%s]' " % config["machine"]["gpu-model"]
            
            if "gpus-per-task" in config["machine"]["options"]:
                sbatch += "'rusage[ngpus_excl_p=%s]' " % config["machine"]["gpus-per-task"]
            
        
    expdir = join(group_root, name)
    log_dir = join(expdir, "logs")
    continuing = False
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
        
    expdir = join(group_root, name)
    if not continuing:
        if args.grid != expdir + ".json":
            shutil.copyfile(args.grid, join(expdir, "sweep.json"))

    # List config of parameters
    paramset = enumerateParams(config["params"])
    param_names = list(set([k for d in paramset for k in d.keys()]))
    param_values = {k: sorted(list(set([d[k] for d in paramset if k in d]))) for k in param_names}


    """if os.path.exists(args.jobnames):
        jobnames = loadlist(args.jobnames)
        jobnames.append("%s/%s" % (group, name))
        savelist(args.jobnames, jobnames)
    """
    if args.launch_parallel:
        launchfilename = tempfile.NamedTemporaryFile().name
        launchfile = open(launchfilename, "w")
        
        
    params_to_index = [k for k, values in param_values.items() if len(values) >= 2 and (
                any(["/" in v for v in values if type(v) is str]) or any([len(str(v)) >= 20 for v in values]))]

    if args.array:
        filename = join(expdir, "run.sh")
        log_stdout = join(log_dir, "common.stdout")
        log_stderr = join(log_dir, "common.stderr")
        with open(filename, "w") as f:
            f.write(cmdPre(config, None, name, log_stdout, log_stderr, filename, num_expes=len(paramset), pooling=args.pooling))

        with open(join(expdir, "params.txt"), "w") as f, open(join(expdir, "commands.txt"), "w") as f_cmd:
            for params in paramset:
                ext = generateExt(params, param_values, to_index=params_to_index)
                params = replaceMacros(params)

                log_stdout = join(log_dir, ext + ".stdout")
                log_stderr = join(log_dir, ext + ".stderr")

                if args.dest_arg:
                    if os.path.exists(join(expdir, ext)):
                        print('WARNING: %s already exists (whatever is in there will probably be overwritten by experiment)' % join(expdir, ext))
                    else:
                        os.makedirs(join(expdir, ext))
                    dest_name = config["meta"]["dest-name"] if "dest-name" in config["meta"] else "dest"
                    dest_name = [dest_name] if type(dest_name) is str else dest_name
                    for dname in dest_name:
                        params[dname] = join(expdir, ext)

                linear_params = linearize_params(params)

                f.write(f"{ext}\t{log_stdout}\t{log_stderr}\t{linear_params}\n")
                f_cmd.write(f"{config['cmd']} {linear_params}\n")

        # print("To launch, execute: ")
        n_chunks = int(ceil(len(paramset) / args.pooling))
        cmd = sbatch + f"-J 'testjobs=[1-{n_chunks}]' {filename}"
        print(cmd)

        subprocess.check_output(["chmod +x %s" % filename], shell=True)
        
        if args.launch:
            r = subprocess.check_output([cmd], shell=True)
            print(r)
        else:
            print("Chmoding %s" % group_root)
            subprocess.check_output(["chmod -R a+w %s" % expdir], shell=True)
    else:
        for i_param, params in enumerate(paramset):
            ext = generateExt(params, param_values, to_index=params_to_index)
            if args.numeric:
                new_ext = "%d" % i_param
                with open(join(log_dir, new_ext + "_params.txt"), "w") as f:
                    f.write(ext)
                ext = new_ext

            params = replaceMacros(params)

            log_stdout = join(log_dir, ext + ".stdout")
            log_stderr = join(log_dir, ext + ".stderr")

            if args.dest_arg:
                if os.path.exists(join(expdir, ext)):
                    print('WARNING: %s already exists (whatever is in there will probably be overwritten by experiment)' % join(expdir, ext))
                else:
                    os.makedirs(join(expdir, ext))
                dest_name = config["meta"]["dest-name"] if "dest-name" in config["meta"] else "dest"
                dest_name = [dest_name] if type(dest_name) is str else dest_name
                for dname in dest_name:
                    params[dname] = join(expdir, ext)

            filename = join(expdir, 'run%s.sh' % ext)
            with open(filename, 'w') as f:
                f.write(cmdPre(config, params, name+ext, log_stdout, log_stderr, filename))
                
            subprocess.check_output(["chmod +x %s" % filename], shell=True)
            
            if args.launch_parallel:
                launchfile.write(sbatch + filename + "&\n")
            else:
                print([sbatch + filename])
                start = time.time()
                if args.launch:
                    r = subprocess.check_output([sbatch + filename], shell=True)
                    # jobid = int(r.rstrip().split(" ")[3])
                    print(r)
                print("Took %.2f" % (time.time() - start))
                print(log_stdout)
                

        if not args.launch:
            print("Chmoding %s" % group_root)
            subprocess.check_output("chmod -R a+w %s" % group_root)

        if args.launch_parallel:
            launchfile.write("wait")
            launchfile.close()
            if args.launch:
                r = subprocess.check_output("bash %s" % launchfilename, shell=True)
                
                
elif args.command == 'count':
    config = json.load(open(args.grid), object_pairs_hook=OrderedDict)
    paramset = enumerateParams(config["params"])

    print("There are %d jobs in this sweep" % len(paramset))
elif args.command == 'remove':
    trash_expe(args.expe, args.trash_dir)
elif args.command == "list":
    print(run_command("ls " + join(args.ckpt_root, args.group)))
# Shamelessly copy-pasted from stool
elif args.command == 'stress':
    if args.partition != "":
        args.partition = " --partition=%s " % args.partition
    stool_stress(args.partition)
elif args.command == "status":
    users = [os.environ.get('USER')]
    users = ','.join(users)
    print(run_command('bqueues -u ' + users + ' -o "%90j | %9A | %4t | %12b | %4C | %10m | %10M | %N | %P | %u"'))
elif args.command == "notebook":
    assert os.path.exists(args.json_path)
    assert not os.path.exists(args.nb_path)

    pass

                
