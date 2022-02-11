#!/bin/bash

#SBATCH --time=600
#SBATCH --gpus-per-task=2
#SBATCH --mem=5000
#SBATCH --output=/cluster/home/kamara/Explain/checkpoints/topk/logs/_explainer_name=pagerank_topk=20_dataset=syn5.stdout
#SBATCH --error=/cluster/home/kamara/Explain/checkpoints/topk/logs/_explainer_name=pagerank_topk=20_dataset=syn5.stderr
#SBATCH --job-name=topk_explainer_name=pagerank_topk=20_dataset=syn5
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

cd .
EXP_NUMBER=$SLURM_ARRAY_TASK_ID
export JOBNAME="topk_explainer_name=pagerank_topk=20_dataset=syn5"
LOG_STDOUT="/cluster/home/kamara/Explain/checkpoints/topk/logs/_explainer_name=pagerank_topk=20_dataset=syn5.stdout"
LOG_STDERR="/cluster/home/kamara/Explain/checkpoints/topk/logs/_explainer_name=pagerank_topk=20_dataset=syn5.stderr"

trap_handler () {
   echo "Caught signal" >> $LOG_STDOUT
   sbatch --begin=now+120 /cluster/home/kamara/Explain/checkpoints/topk/run_explainer_name=pagerank_topk=20_dataset=syn5.sh
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
--explainer_name pagerank --topk 20 --dataset syn5 --num_test_nodes 100 --data_save_dir data --gpu True --dest /cluster/home/kamara/Explain/checkpoints/topk/_explainer_name=pagerank_topk=20_dataset=syn5 >> $LOG_STDOUT 2>> $LOG_STDERR && echo 'JOB_FINISHED' >> $LOG_STDOUT &
wait $!
