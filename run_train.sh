#!/usr/bin/env bash
DATE=`date '+%Y-%m-%d_%H-%M'`
TRAIN_PATH="/usr/local/share/models/"
TRAIN_DIR=$TRAIN_PATH$DATE
#TRAIN_DIR="/usr/local/share/models/2018-08-11_11-52/"

if [[ ! -e $DATA_DIR ]]; then
    echo "Data dir $DATA_DIR does not exists." 1>&2
    exit 1
fi
if [[ ! -e $TRAIN_DIR ]]; then
    mkdir $TRAIN_DIR
elif [[ ! -d $TRAIN_DIR ]]; then
    echo "Model dir $TRAIN_DIR already exists but is not a directory" 1>&2
fi

PYTHONPATH="./" python train.py \
--model_dir $TRAIN_DIR \
--configuration "v_1_0_SqNxt_23_mod" \
--batch_size 64 \
--num_epochs_per_length 5 \
--sequence_length 2 \
--num_sequence_lengths 5 \
--fine_tune_ckpt "/usr/local/share/models/squeezenext_mod/" \
--training_file_pattern $DATA_DIR"train-*.tfrecord" \
--validation_file_pattern $DATA_DIR"val-*.tfrecord"
