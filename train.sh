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
memory_length=$4
latent_Actor=$5

exp_name=${model}'-'${task}'-'${seed}'-'${memory_length}


nohup python src/main.py env.train.id=${task} world_model.model=${model} world_model.wm_memory_length=${memory_length} wandb.name=${exp_name} common.seed=${seed} \
> 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'