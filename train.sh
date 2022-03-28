#!/bin/bash

START=$1
END=$2

for ((i=START;i<=END;i++)); do
    printf "\n\n${i}\n\n"
    python train.py model=taco_net experiment_name=test vs_type=jet dataset_name=SM dataset_id=$i n_epochs=1
done

