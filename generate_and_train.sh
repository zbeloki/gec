# NAME
# SENTS
# DS
SEED=65

python3 generate_synthetic_data.py $SENTS $CORRUPTION/$DS/type_classifier $CORRUPTION/$DS/span_classifier $CORRUPTION/$DS/esg_model $TRAIN_WI_M2 $SYNTH/$NAME.csv $SEED;

DEV=$DEV_WI_M2 bash train.sh $SYNTH/$NAME.csv $CORRECTION/$NAME
