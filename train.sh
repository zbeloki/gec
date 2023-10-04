# Script interface
: '
Positional arguments:
[1] TRAIN_FILE
[2] MODEL_OUTPUT

Required environment variables:
- DEV
- CUDA_VISIBLE_DEVICES

Optional environment variables:
- INPUT_MODEL
- BATCH_SIZE
- EPOCHS
- LR
- EVAL_STEPS
- SEED
'

# Positional arguments

TRAIN_FILE=$1
MODEL_OUTPUT=$2

if [ -z ${TRAIN_FILE} ] || [ -z ${MODEL_OUTPUT} ]; then
    echo "Usage: `basename "$0"` TRAIN_FILE MODEL_OUTPUT" 1>&2
    exit 1
fi

# Required environment variables

if [ -z ${DEV} ]; then
    echo "ERROR: You must set the environment variable DEV" 1>&2
    exit 1
fi
if [ -z ${CUDA_VISIBLE_DEVICES} ]; then
    echo "ERROR: You must set the environment variable CUDA_VISIBLE_DEVICES to choose the GPU to use" 1>&2
    exit 1
fi

# Optional environment variables

DEF_INPUT_MODEL="facebook/bart-base"
DEF_BATCH_SIZE=16
DEF_EPOCHS=3
DEF_SEED=42

if [ -z ${INPUT_MODEL} ]; then
    INPUT_MODEL=$DEF_INPUT_MODEL
fi
if [ -z ${BATCH_SIZE} ]; then
    BATCH_SIZE=$DEF_BATCH_SIZE
fi
if [ -z ${EPOCHS} ]; then
    EPOCHS=$DEF_EPOCHS
fi
if [ -z ${SEED} ]; then
    SEED=$DEF_SEED
fi

LOAD_BEST=True
EVAL_STRATEGY=epoch
DEF_EVAL_STEPS=1
if [ -z ${EVAL_STEPS} ]; then
    EVAL_STEPS=$DEF_EVAL_STEPS
else
    LOAD_BEST=False
    EVAL_STRATEGY=steps
fi

# Constant parameters

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
TRAIN_SCRIPT=$DIR/finetune_seq2seq.py

# Select metric_for_best_model
DEV_EXT="${DEV##*.}"
if [ "$DEV_EXT" == "csv" ];
then
    METRIC="eval_rouge1"
else
    METRIC="eval_m2_f0.5"
fi
echo "Metric for best model: $METRIC"

# Execution

python3 $TRAIN_SCRIPT \
	--model_name_or_path $INPUT_MODEL \
	--output_dir $MODEL_OUTPUT \
	--overwrite_output_dir \
	--do_train \
	--train_file $TRAIN_FILE \
	--do_eval \
	--validation_file $DEV \
	--gradient_accumulation_steps 2 \
	--per_device_train_batch_size=$BATCH_SIZE \
	--per_device_eval_batch_size=16 \
	--num_train_epochs $EPOCHS \
	--predict_with_generate \
	--load_best_model_at_end=$LOAD_BEST \
	--save_total_limit=1 \
	--save_strategy epoch \
	--evaluation_strategy $EVAL_STRATEGY \
	--eval_steps $EVAL_STEPS \
	--metric_for_best_model $METRIC \
	--use_fast false \
	--max_source_length 256 \
	--max_target_length 256 \
	--generation_max_length 256 \
	--seed $SEED

# Predict Wi test
bash script/predict_wi_test.sh $MODEL_OUTPUT
