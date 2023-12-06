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
reg_tokens=$6
reg_emb=$7
reg_post_quant=$8
reg_consistency=$9


exp_name=${model}'-'${task}'-'${seed}'-linear_actor:'${linear_actor}'-emb_inpt:'${embedding_input}'-reg_tokens:'${reg_tokens}'-reg_emb:'${reg_emb}'-reg_post:'${reg_post_quant}'-reg_consistency:'${reg_consistency}


nohup python src/main.py \
env.train.id=${task} \
world_model.model=${model} \
world_model.regularization_tokens=${reg_tokens} \
world_model.regularization_embeddings=${reg_emb} \
world_model.regularization_post_quant=${reg_post_quant} \
world_model.embedding_input=${embedding_input} \
tokenizer.consistency_loss_reg=${reg_consistency} \
actor_critic.linear_actor=${linear_actor} \
wandb.name=${exp_name} \
common.seed=${seed} \
> 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'