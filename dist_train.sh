#!/usr/bin/env bash

GPUS=$1
CONFIG=$2
PORT=$3
# usage
#if [ $# -ne 2 ] ;then
#    echo "usage:"
#    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
#    exit
#fi

echo $PORT
#PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    ./train.py -opt $CONFIG --launcher pytorch
