#!/bin/bash

for ((i=0;i<3;i+=1))
do
	python main.py \ 
        --algorithm="ddpg" \
        --n_timesteps=1000000 \
        --seed=$i
	
    python main.py \ 
        --algorithm="td3" \
        --n_timesteps=1000000 \
        --seed=$i
done