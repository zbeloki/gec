import m2

import datasets
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer

import itertools
from collections import Counter
import pdb
import sys
import argparse

MODEL_ID_EN = "bert-base-cased"
MODEL_ID_EU = "orai-nlp/ElhBERTeu"

def main(train_fpath, dev_fpath, use_simple_types, out_path, lang):
  
  def load_dataset_from_m2_files(m2_fpaths):
    m2_entries = []
    for m2_fpath in m2_fpaths:
      with open(m2_fpath, 'r') as f:
        for m2e in m2.read_m2(f):
          m2_entries.append({
              'sentence': m2e.original(),
              'labels': [ m2.simplify_type(a.error_type) if use_simple_types else a.error_type
                          for a in m2e.annotations(ignore_unk=True, ignore_empty=True) ],
          })
    ds = datasets.Dataset.from_list(m2_entries)
    return ds

  dss = datasets.DatasetDict({
      'train': load_dataset_from_m2_files([train_fpath]),
      'test': load_dataset_from_m2_files([dev_fpath]),
  })

  error_types = m2.eu_error_types() if lang == 'eu' else m2.error_types(simplified=use_simple_types)

  mlb = MultiLabelBinarizer()
  mlb.fit([error_types])

  def one_hot_encode(batch):
      batch['labels'] = mlb.transform(batch['labels'])
      return batch
  dss = dss.map(one_hot_encode, batched=True)
  dss = dss.cast_column('labels', datasets.Sequence(datasets.Value(dtype='float32', id=None)))

  print(f"Number of types: {len(error_types)}")
  print(f"Sentences in train set: {len(dss['train'])}")

  model_id = MODEL_ID_EU if lang == 'eu' else MODEL_ID_EN
  tokenizer = AutoTokenizer.from_pretrained(model_id)

  tokenize = lambda batch: tokenizer(batch['sentence'], truncation=True, padding=True, max_length=256)
  dss_tok = dss.map(tokenize, batched=True, batch_size=None)

  BATCH_SIZE = 32
  GRAD_ACC = 1
  N_EPOCHS = 6
  DEVICE = 'cuda'

  def model_init():
    id2label = { i:mlb.classes_[i] for i in range(len(mlb.classes_)) }
    model = AutoModelForSequenceClassification.from_pretrained(model_id, id2label=id2label)
    model.config.problem_type = "multi_label_classification"
    return model

  metric_p = datasets.load_metric("precision", config_name='multilabel')
  metric_r = datasets.load_metric("recall", config_name='multilabel')
  metric_f = datasets.load_metric("f1", config_name='multilabel')

  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probas = sigmoid(logits)
    preds = (probas >= 0.5).astype(int)
    metrics = {
      'precision': metric_p.compute(predictions=preds,
                                    references=labels,
                                    average='micro')['precision'],
      'recall': metric_r.compute(predictions=preds,
                                  references=labels,
                                  average='micro')['recall'],
      'f1': metric_f.compute(predictions=preds,
                              references=labels,
                              average='micro')['f1'],
    }
    return metrics

  args = TrainingArguments(output_dir=out_path,
                           metric_for_best_model="f1",
                           num_train_epochs=N_EPOCHS,
                           per_device_train_batch_size=int(BATCH_SIZE/GRAD_ACC),
                           gradient_accumulation_steps=GRAD_ACC,
                           per_device_eval_batch_size=BATCH_SIZE,
                           evaluation_strategy='epoch',
                           save_strategy='epoch',
                           load_best_model_at_end=True,
                           disable_tqdm=False,
                           log_level='error',
                           logging_steps=100)

  trainer = Trainer(model_init=model_init,
                    args=args,
                    compute_metrics=compute_metrics,
                    train_dataset=dss_tok['train'],
                    eval_dataset=dss_tok['test'],
                    tokenizer=tokenizer)

  trainer.train()

  trainer.save_model()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("TRAIN_M2", help='M2 file containing the training data')
    parser.add_argument("DEV_M2", help='M2 file containing the dev data')
    parser.add_argument("OUTPUT_MODEL", help='The model will be created in this path')
    parser.add_argument("--simple-types", action='store_true', help='Use simplified types (24) or original types (54)')
    parser.add_argument("--lang", default='en', choices=['en', 'eu'], help='Lang for base model: en, eu')
    args = parser.parse_args()
    
    main(args.TRAIN_M2, args.DEV_M2, args.simple_types, args.OUTPUT_MODEL, args.lang)
