#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 OUT_FILE"
    exit 1
fi
OUT=$1

python3 generate_synthetic_data.py \
	--sentences /mnt/nfs/gec/en/sources_for_synthetic_data/wmt_news/wmt2021/wmt.35k.txt \
	--type_classifier models/corruption/wi/type_classifier \
	--span_classifier models/corruption/wi/span_classifier \
	--esg_model models/corruption/wi/esg_model \
	--ref_m2 /mnt/nfs/gec/en/original_datasets/wi+locness/m2/ABC.train.gold.bea19.m2 \
	--out $OUT
	#--force_single_error
