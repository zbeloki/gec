import utils
import m2

import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForQuestionAnswering
from datasets import Dataset, DatasetDict
import numpy as np

import argparse
import pdb

MODEL_ID_EN = "bert-base-cased"
MODEL_ID_EU = "orai-nlp/ElhBERTeu"
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
GRAD_ACC = 1

def main(train_m2_fpath, dev_m2_fpath, use_simple_types, out_path, lang):

    error_types = m2.eu_error_types() if lang == 'eu' else m2.error_types(simplified=use_simple_types)
    print(f"Working with {len(error_types)} error types (check --simple-types argument)")

    model_id = MODEL_ID_EU if lang == 'eu' else MODEL_ID_EN
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_tokens(error_types)

    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    data = DatasetDict({
        'train': utils.m2_to_dataset(train_m2_fpath, use_simple_types=use_simple_types, invert=True),
        'dev': utils.m2_to_dataset(dev_m2_fpath, use_simple_types=use_simple_types, invert=True),
    })
    data = data.map(
        lambda examples: preprocess_dataset(examples, tokenizer),
        batched=True,
        remove_columns=data["train"].column_names
    )

    def compute_metrics(eval_preds):
        (start_logits, end_logits), true_labels = eval_preds
        n_examples = len(start_logits)

        match = 0
        overlap = 0
        manual = 0
        for i in range(n_examples):
            pred_span = utils.infer_token_span(start_logits[i], end_logits[i])
            true_span = (true_labels[0][i], true_labels[1][i])

            if pred_span[0] == true_span[0] and pred_span[1] == true_span[1]:
                match += 1
            if pred_span[0] < true_span[1] and pred_span[1] > true_span[0]:
                overlap += 1
            if pred_span[2] == 0.0:
                manual += 1

        return {
            'overlap': overlap / n_examples,
            'match': match / n_examples,
            'span_found': 1 - (manual/n_examples),
        }

    args = TrainingArguments(
        output_dir=out_path,
        metric_for_best_model='match',
        num_train_epochs=4,
        per_device_train_batch_size=int(BATCH_SIZE/GRAD_ACC),
        gradient_accumulation_steps=GRAD_ACC,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy='epoch',
        #eval_steps=100,
        save_strategy="epoch",
        #save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        disable_tqdm=False,
        log_level='error',
        fp16=True,
        logging_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    trainer.save_model()


def preprocess_dataset(examples, tokenizer):
    inputs = tokenizer(examples['error_type'], examples['sentence'], max_length=MAX_SEQ_LEN,
                       truncation='only_second', return_offsets_mapping=True, padding='max_length')
    start_toks = []
    end_toks = []
    for i, offsets in enumerate(inputs['offset_mapping']):
        seq_ids = inputs.sequence_ids(i)
        try:
            context_start = seq_ids.index(1)
        except Exception as e:
            # context is empty
            start_toks.append(len(seq_ids))
            end_toks.append(len(seq_ids))
            continue
        context_end = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        ans_start_char, ans_end_char = examples['span'][i]

        idx = context_start
        while idx < context_end and offsets[idx+1][0] <= ans_start_char:
            idx += 1
        ans_start_tok = idx
        if idx == context_end and ans_start_char >= offsets[context_end][1]:
            # when the span starts after the last token of the sentence
            ans_start_tok = context_end + 1
        start_toks.append(ans_start_tok)

        while idx <= context_end and offsets[idx][1] <= ans_end_char:
            idx += 1
        end_toks.append(idx)
    inputs['start_positions'] = start_toks
    inputs['end_positions'] = end_toks
    return inputs


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("TRAIN_M2", help='M2 file containing the training data')
    parser.add_argument("DEV_M2", help='M2 file containing the dev data')
    parser.add_argument("OUTPUT_MODEL", help='The model will be created in this path')
    parser.add_argument("--simple-types", action='store_true', help='Use simplified types (24) or original types (54)')
    parser.add_argument("--lang", default='en', choices=['en', 'eu'], help='Lang for base model: en, eu')
    args = parser.parse_args()
    
    main(args.TRAIN_M2, args.DEV_M2, args.simple_types, args.OUTPUT_MODEL, args.lang)
