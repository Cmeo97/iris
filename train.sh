#!/bin/bash

conda activate iris

task=BoxingNoFrameskip-v4 

seed=0


exp_name='irisXL-discrete-'${task}'-'${seed}


nohup python src/main.py env.train.id=${task} wandb.name=${exp_name} common.seed=${seed} \
> 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'