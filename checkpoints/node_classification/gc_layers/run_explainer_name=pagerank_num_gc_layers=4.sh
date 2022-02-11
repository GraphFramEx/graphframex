#!/bin/bash

#SBATCH --gpus-per-task=8
#SBATCH --mem=4000
#SBATCH --gpu_model0=GeForceGTX1080Ti
#SBATCH --output=/cluster/home/kamara/Explain/checkpoints/gc_layers/logs/_explainer_name=pagerank_num_gc_layers=4.stdout
#SBATCH --error=/cluster/home/kamara/Explain/checkpoints/gc_layers/logs/_explainer_name=pagerank_num_gc_layers=4.stderr
#SBATCH --job-name=gc_layers_explainer_name=pagerank_num_gc_layers=4
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

cd .
EXP_NUMBER=$SLURM_ARRAY_TASK_ID
export JOBNAME="gc_layers_explainer_name=pagerank_num_gc_layers=4"
LOG_STDOUT="/cluster/home/kamara/Explain/checkpoints/gc_layers/logs/_explainer_name=pagerank_num_gc_layers=4.stdout"
LOG_STDERR="/cluster/home/kamara/Explain/checkpoints/gc_layers/logs/_explainer_name=pagerank_num_gc_layers=4.stderr"

trap_handler () {
   echo "Caught signal" >> $LOG_STDOUT
   sbatch --begin=now+120 /cluster/home/kamara/Explain/checkpoints/gc_layers/run_explainer_name=pagerank_num_gc_layers=4.sh
   exit 0
}
function ignore {
   echo "Ignored SIGTERM" >> $LOG_STDOUT
}

trap ignore TERM
trap trap_handler USR1
echo "Git hash:" >> $LOG_STDOUT
echo $(git rev-parse HEAD 2> /dev/null) >> $LOG_STDOUT

which python >> $LOG_STDOUT
echo "---Beginning program ---" >> $LOG_STDOUT
PYTHONUNBUFFERED=yes MKL_THREADING_LAYER=GNU python exp_synthetic/main.py \
--explainer_name pagerank --num_gc_layers 4 --dataset syn1 --num_test_nodes 200 --data_save_dir data --dest /cluster/home/kamara/Explain/checkpoints/gc_layers/_explainer_name=pagerank_num_gc_layers=4 >> $LOG_STDOUT 2>> $LOG_STDERR && echo 'JOB_FINISHED' >> $LOG_STDOUT &
wait $!
