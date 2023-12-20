#!/bin/bash
#SBATCH --job-name=IRIS
#SBATCH --partition=long                      
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=45G                                  
#SBATCH --time=6-20:00:00

conda activate irisXL

#task=BoxingNoFrameskip-v4 

model=$1
task=$2
seed=$3
linear_actor=$4
embedding_input=$5


exp_name=${model}'-'${task}'-'${seed}'-linear_actor:'${linear_actor}'-emb_inpt:'${embedding_input}


nohup python src/main.py \
env.train.id=${task} \
world_model.model=${model} \
world_model.embedding_input=${embedding_input} \
actor_critic.linear_actor=${linear_actor} \
wandb.name=${exp_name} \
common.seed=${seed} \
> 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'