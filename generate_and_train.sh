# NAME
# SENTS
SEED=59

python3 generate_synthetic_data.py $SENTS $CORRUPTION/wi/type_classifier $CORRUPTION/wi/span_classifier $CORRUPTION/wi/esg_model $TRAIN_WI_M2 $SYNTH/$NAME.csv $SEED;

DEV=$DEV_WI_M2 bash train.sh $SYNTH/$NAME.csv $CORRECTION/$NAME
