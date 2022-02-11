#!/bin/bash

#SBATCH --time=600
#SBATCH --gpus-per-task=2
#SBATCH --mem=5000
#SBATCH --output=/cluster/home/kamara/Explain/checkpoints/node_classification/sparsity_mask/logs/_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False.stdout
#SBATCH --error=/cluster/home/kamara/Explain/checkpoints/node_classification/sparsity_mask/logs/_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False.stderr
#SBATCH --job-name=sparsity_mask_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

cd .
EXP_NUMBER=$SLURM_ARRAY_TASK_ID
export JOBNAME="sparsity_mask_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False"
LOG_STDOUT="/cluster/home/kamara/Explain/checkpoints/node_classification/sparsity_mask/logs/_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False.stdout"
LOG_STDERR="/cluster/home/kamara/Explain/checkpoints/node_classification/sparsity_mask/logs/_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False.stderr"

trap_handler () {
   echo "Caught signal" >> $LOG_STDOUT
   sbatch --begin=now+120 /cluster/home/kamara/Explain/checkpoints/node_classification/sparsity_mask/run_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False.sh
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
PYTHONUNBUFFERED=yes MKL_THREADING_LAYER=GNU python code/main.py \
--explainer_name sa --sparsity 0.99 --dataset syn1 --hard_mask False --explain_graph False --num_test 100 --data_save_dir data --dest /cluster/home/kamara/Explain/checkpoints/node_classification/sparsity_mask/_explainer_name=sa_sparsity=0.99_dataset=syn1_hard_mask=False >> $LOG_STDOUT 2>> $LOG_STDERR && echo 'JOB_FINISHED' >> $LOG_STDOUT &
wait $!
