MODEL=$1
DEV_FILE=$DEV # env-var

if [ -z $DEV ]; then
    echo "Environment variable DEV undefined"
    exit
fi

DIR_TMP=`mktemp -d`
SENTS_TMP=$DIR_TMP/sents.txt
PREDS_TMP=$DIR_TMP/sents.pred.txt
PRED_M2_TMP=$DIR_TMP/pred.m2

grep -P "^S " $DEV_FILE | cut -c 3- > $SENTS_TMP
python3 predict.py $MODEL $SENTS_TMP $PREDS_TMP
errant_parallel -orig $SENTS_TMP -cor $PREDS_TMP -out $PRED_M2_TMP
errant_compare -hyp $PRED_M2_TMP -ref $DEV_FILE
#$ERRANT_BIN/errant_compare -hyp $PRED_M2_TMP -ref $DEV_FILE -cat 1

rm $SENTS_TMP $PREDS_TMP $PRED_M2_TMP
rm -rf $DIR_TMP
