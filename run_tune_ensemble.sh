#!/bin/bash
# Read a string with spaces using for loop
# Use this for hyperparameter tuning.
dropout=('0.1' '0.2' '0.4' '0.6' '0.8' '0.9')
lrclinical=('0.01' '0.02' '0.03' '0.04' '0.05' '0.06')
lrmri=('0.001' '0.002' '0.003' '0.004' '0.005' '0.006')
mode=$1


for drop in "${dropout[@]}"   ### Outer for loop ###
do
    for lrclin in "${lrclinical[@]}" ### Inner for loop 1 ###
    do
      for lramri in "${lrmri[@]}" ### Inner for loop 2 ###
      do
        python3 train_resnet_ensemble.py --dropout "$drop" --lr_clinical "$lrclin" --lr_mri "$lramri" --normalization --model_type "$mode"
      done
    done
done