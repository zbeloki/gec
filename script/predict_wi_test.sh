#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 MODEL_PATH"
    exit 1
fi
MODEL=$1

TEST_FNAME=`basename $TEST_WI_TXT`

python3 predict.py $MODEL $TEST_WI_TXT $MODEL/$TEST_FNAME.pred.txt

zip -j $MODEL/$TEST_FNAME.pred.txt.zip $MODEL/$TEST_FNAME.pred.txt
