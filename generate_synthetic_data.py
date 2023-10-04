import m2
import utils
from tokenizer import Tokenizer

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForQuestionAnswering, BartForConditionalGeneration
import numpy as np
import tqdm

from collections import Counter
import logging
logging.basicConfig(level=logging.INFO)
import random
import math
import argparse
import csv
import pdb

REF_WEIGHT = 0.5
ERROR_EACH_N_TOKENS = 10
DEF_SEED = 42
BATCH_SIZE = 16

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

def main(sent_fpath, type_cls_path, rnd_types, span_cls_path, esg_path, ref_fpath, seed, single_error, out_fpath):

    # reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Using seed: {seed}")

    if single_error:
        logging.info("Forcing to create a single error on each example")
        
    tokenizer = Tokenizer()

    # type classifier
    logging.info("Loading type classifier model")
    type_tokenizer = AutoTokenizer.from_pretrained(type_cls_path)
    type_model = AutoModelForSequenceClassification.from_pretrained(type_cls_path).to(device)
    error_types = list(type_model.config.id2label.values())
    if rnd_types:
        logging.info("Random type selection activated")

    # span classifier
    if span_cls_path is not None:
        logging.info("Loading span classifier model")
        span_tokenizer = AutoTokenizer.from_pretrained(span_cls_path)
        span_tokenizer.add_tokens(error_types)
        span_model = AutoModelForQuestionAnswering.from_pretrained(span_cls_path).to(device)
        span_model.resize_token_embeddings(len(span_tokenizer))
        span_based_esg = True
    else:
        logging.info("Non-span based ESG activated")
        span_based_esg = False

    # error-span generation model
    logging.info("Loading error-span generation model")
    esg_tokenizer = AutoTokenizer.from_pretrained(esg_path)
    esg_tokenizer.add_tokens([utils.SEP_TOKEN, utils.MASK_TOKEN])
    esg_model = BartForConditionalGeneration.from_pretrained(esg_path)
    esg_model.resize_token_embeddings(len(esg_tokenizer))
    esg_pipe = pipeline('text2text-generation', max_length=64, tokenizer=esg_tokenizer, model=esg_model, device=device)

    # load reference type counts
    use_simple_types = error_types[0][:2] not in ['R:', 'U:', 'M:']
    if use_simple_types:
        logging.info("Using simplified types")
    else:
        logging.info("Using original (R:M:U) types")
    ref_type_probs = utils.load_type_rates(ref_fpath, use_simple_types)
        
    logging.info("Generating synthetic data")
    sents = [ ln.strip() for ln in open(sent_fpath, 'r') if 3 < len(ln.strip().split()) < 50 ]
    sents.sort(key=lambda s: len(s.split()), reverse=True)
    num_sents = len(sents)
    with open(out_fpath, 'w') as f_out, \
         open(out_fpath+'.types.txt', 'w') as f_types:
        writer = csv.DictWriter(f_out, fieldnames=['source', 'target'])
        writer.writeheader()
        for batch in tqdm.tqdm(SentenceBatches(sents, BATCH_SIZE), total=math.ceil(num_sents/BATCH_SIZE)):
            orig_sents = batch
            sents = [ s for s in orig_sents ]
            n_errors = get_n_errors(sents[0], tokenizer) if not single_error else 1
            used_error_types = [ set() for _ in range(BATCH_SIZE) ]
            for _ in range(n_errors):
                if rnd_types:
                    error_type = random.sample(error_types, BATCH_SIZE)
                else:
                    error_type = get_error_type(sents, type_tokenizer, type_model, ref_type_probs, used_error_types)
                if span_based_esg:
                    span = get_span(sents, error_type, span_tokenizer, span_model)
                else:
                    span = [ (0, len(sent)) for sent in sents ]
                error_text = get_error_text(sents, error_type, span, esg_pipe)
                sents = [ (sent[:span[i][0]] + " " + error_text[i] + " " + sent[span[i][1]:]) for i, sent in enumerate(sents) ]
                sents = [ tokenizer.detokenize(sent) for sent in sents ]
                for i in range(len(used_error_types)):
                    used_error_types[i].add(error_type[i])
                    print(error_type[i], file=f_types)
            for i in range(len(sents)):
                writer.writerow({
                    'source': sents[i],
                    'target': orig_sents[i],
                })
                logging.debug(f"ORIG: {orig_sents[i]}")
                logging.debug(f"SYNT: {sents[i]} <- {error_type[i]}")


class SentenceBatches:
    def __init__(self, sents, size):
        self.sents = sents
        self.size = size
        self.i = 0
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i*self.size >= len(self.sents):
            raise StopIteration
        batch = self.sents[self.i*self.size : (self.i+1)*self.size]
        self.i += 1
        return batch

def get_n_errors(sent, tokenizer):
    MAX_ERRORS = 8
    tokens = tokenizer.tokenize(sent)
    n_errors = math.ceil(len(tokens) / ERROR_EACH_N_TOKENS)
    n_errors = min(n_errors, MAX_ERRORS)
    return n_errors

def get_error_type(sents, tokenizer, model, ref_probs, used_types):
    inputs = tokenizer(sents, max_length=utils.MAX_SEQ_LEN, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**inputs)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probs = sigmoid(output.logits.cpu().numpy())
    error_types = []
    for i in range(len(probs)):
        inferred_probs = { model.config.id2label[tid]: probs[i][tid] for tid in range(len(probs[i])) }
        error_type = select_type(inferred_probs, ref_probs, used_types[i])
        error_types.append(error_type)
    return error_types

def select_type(inferred_probs, ref_probs, used_types):
    types = ref_probs.keys()
    error_types = []
    probs = { t: inferred_probs[t] * (1-REF_WEIGHT) + ref_probs[t] * REF_WEIGHT for t in types }
    for error_type in used_types:
        probs[error_type] = 0.0
    probs = { t: probs[t]/sum(probs.values()) for t in types }
    error_type = np.random.choice(list(probs.keys()), p=list(probs.values()))
    return error_type

def get_span(sents, error_types, tokenizer, model):
    inputs = tokenizer(error_types, sents, max_length=utils.MAX_SEQ_LEN, return_tensors='pt',
                       truncation='only_second', return_offsets_mapping=True, padding='max_length')
    offset_mappings = inputs.pop('offset_mapping').tolist()
    model_inputs = { col: inputs[col].to(model.device) for col in inputs.keys() }
    with torch.no_grad():
        outputs = model(**model_inputs)
    batch_start_logits = outputs.start_logits.cpu().numpy()
    batch_end_logits = outputs.end_logits.cpu().numpy()
    spans = []
    for i in range(len(batch_start_logits)):
        span = utils.infer_char_span(batch_start_logits[i], batch_end_logits[i], offset_mappings[i], inputs.word_ids(i))
        spans.append(span)
    return [ (span[0], span[1]) for span in spans ]

def get_error_text(sents, error_types, char_spans, pipeline):
    esg_inputs = [ utils.to_esg_input_format(sents[i], error_types[i], char_spans[i]) for i in range(len(sents)) ]
    esg_output = pipeline(esg_inputs, truncation=True, batch_size=BATCH_SIZE, num_return_sequences=2)
    error_texts = []
    for i in range(len(sents)):
        orig_text = sents[i][char_spans[i][0]:char_spans[i][1]]
        new_text = esg_output[i][0]['generated_text'].strip()
        if new_text == orig_text.strip():
            new_text = esg_output[i][1]['generated_text'].strip()
        error_texts.append(new_text)
    return error_texts


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sentences", required=True, help='')
    parser.add_argument("--esg_model", required=True, help='')
    parser.add_argument("--type_classifier", required=True, help='')
    parser.add_argument("--random_types", action='store_true', help='If True, type_classifier is only used to collect the set of error types, but they are randomly picked for each example')
    parser.add_argument("--span_classifier", help='')
    parser.add_argument("--ref_m2", required=True, help='')
    parser.add_argument("--seed", type=int, default=DEF_SEED, help='')
    parser.add_argument("--force_single_error", action='store_true', help='')
    parser.add_argument("--out", required=True, help='Output CSV file')
    args = parser.parse_args()
    
    main(args.sentences, args.type_classifier, args.random_types, args.span_classifier, args.esg_model, args.ref_m2, args.seed, args.force_single_error, args.out)
