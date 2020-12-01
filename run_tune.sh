#!/bin/bash
# Read a string with spaces using for loop.
# Use this for hyperparameter tuning.
normalizations=('zscore' 'whitestripe' 'gmm')
groups=(10 15)
optimizers=('AdamW' 'Adam' 'AdamWR')
patience=$1
mode=$2


for normal in "${normalizations[@]}"   ### Outer for loop ###
do
    for group in "${groups[@]}" ### Inner for loop 1 ###
    do
      for optim in "${optimizers[@]}" ### Inner for loop 2 ###
      do
        python3 train_resnet.py --groups "$group" --optimizer "$optim" --patience "$patience" --normalization "$normal" --model_type "$mode"
      done
    done
done