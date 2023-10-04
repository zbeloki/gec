#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 TRAIN_TSV OUT_DIR"
    exit 1
fi
DATA=$1
OUT=$2

SEED=42 bash train.sh $DATA $OUT/seed42
SEED=13 bash train.sh $DATA $OUT/seed13
SEED=17 bash train.sh $DATA $OUT/seed17
