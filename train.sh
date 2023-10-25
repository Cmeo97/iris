#!/bin/bash

conda activate iris

task=BoxingNoFrameskip-v4 

seed=1


exp_name='vanilla-iris-'${task}'-'${seed}


nohup python src/main.py env.train.id=${task} wandb.name=${exp_name} common.seed=${seed} \
> 'logs/'${exp_name}'.out' 2> 'logs/'${exp_name}'.err'