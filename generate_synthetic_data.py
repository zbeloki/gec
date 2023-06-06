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
import argparse
import csv
import pdb

REF_WEIGHT = 0.5

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

def main(sent_fpath, type_cls_path, span_cls_path, esg_path, ref_fpath, out_fpath):

    detokenizer = Tokenizer(detokenizer_only=True)

    # type classifier
    logging.info("Loading type classifier model")
    type_tokenizer = AutoTokenizer.from_pretrained(type_cls_path)
    type_model = AutoModelForSequenceClassification.from_pretrained(type_cls_path).to(device)
    error_types = list(type_model.config.id2label.values())

    # span classifier
    logging.info("Loading span classifier model")
    span_tokenizer = AutoTokenizer.from_pretrained(span_cls_path)
    span_tokenizer.add_tokens(error_types)
    span_model = AutoModelForQuestionAnswering.from_pretrained(span_cls_path).to(device)
    span_model.resize_token_embeddings(len(span_tokenizer))

    # error-span generation model
    logging.info("Loading error-span generation model")
    esg_tokenizer = AutoTokenizer.from_pretrained(esg_path)
    esg_tokenizer.add_tokens([utils.SEP_TOKEN, utils.MASK_TOKEN])
    esg_model = BartForConditionalGeneration.from_pretrained(esg_path)
    esg_model.resize_token_embeddings(len(esg_tokenizer))
    esg_pipe = pipeline('text2text-generation', max_length=64, tokenizer=esg_tokenizer, model=esg_model, device=device)

    # load reference type counts
    use_simple_types = error_types[0][:2] not in ['R:', 'U:', 'M:']
    ref_type_probs = utils.load_type_rates(ref_fpath, use_simple_types)
        
    logging.info("Generating synthetic data")
    num_sents = sum(1 for ln in open(sent_fpath,'r'))
    with open(sent_fpath, 'r') as f_in, \
         open(out_fpath, 'w') as f_out, \
         open(out_fpath+'.types.txt', 'w') as f_types:
        writer = csv.DictWriter(f_out, fieldnames=['source', 'target'])
        writer.writeheader()
        for ln in tqdm.tqdm(f_in, total=num_sents):
            orig_sent = ln.strip()
            sent = orig_sent
            n_errors = get_n_errors(sent)
            error_types = get_error_types(sent, type_tokenizer, type_model, ref_type_probs, n_errors)
            for error_type in error_types:
                span = get_span(sent, error_type, span_tokenizer, span_model)
                error_text = get_error_text(sent, error_type, span, esg_pipe)
                sent = sent[:span[0]] + " " + error_text + " " + sent[span[1]:]
                sent = detokenizer.detokenize(sent)
                print(error_type, file=f_types)
            writer.writerow({
                'source': sent,
                'target': orig_sent,
            })
            logging.debug(f"ORIG: {orig_sent}")
            logging.debug(f"SYNT: {sent} <- {error_type}")


def get_n_errors(sent):
    #n = np.random.choice([1, 2, 3, 4], p=[0.1, 0.25, 0.4, 0.25])
    return 3

def get_error_types(sent, tokenizer, model, ref_probs, n):

    inputs = tokenizer(sent, max_length=utils.MAX_SEQ_LEN, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        output = model(**inputs)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    probs = sigmoid(output.logits.cpu().numpy())[0]
    inferred_probs = { model.config.id2label[tid]: probs[tid] for tid in range(len(probs)) }
    error_types = select_types(inferred_probs, ref_probs, n)
    return error_types

def select_types(inferred_probs, ref_probs, n):
    types = ref_probs.keys()
    error_types = []
    probs = { t: inferred_probs[t] * (1-REF_WEIGHT) + ref_probs[t] * REF_WEIGHT for t in types }
    for _ in range(n):
        probs = { t: probs[t]/sum(probs.values()) for t in types }
        error_type = np.random.choice(list(probs.keys()), p=list(probs.values()))
        error_types.append(error_type)
        probs[error_type] = 0.0
    return error_types

def get_span(sent, error_type, tokenizer, model):
    inputs = tokenizer(error_type, sent, max_length=utils.MAX_SEQ_LEN, return_tensors='pt',
                       truncation='only_second', return_offsets_mapping=True, padding='max_length')
    offset_mapping = inputs.pop('offset_mapping')[0].tolist()
    model_inputs = { col: inputs[col].to(model.device) for col in inputs.keys() }
    with torch.no_grad():
        outputs = model(**model_inputs)
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()
    span = utils.infer_char_span(start_logits, end_logits, offset_mapping, inputs.word_ids())
    return span[0], span[1]

def get_error_text(sent, error_type, char_span, pipeline):
    esg_input = utils.to_esg_input_format(sent, error_type, char_span)
    esg_output = pipeline(esg_input, truncation=True)
    return esg_output[0]['generated_text'].strip()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("SENTENCES", help='')
    parser.add_argument("TYPE_CLASSIFIER", help='')
    parser.add_argument("SPAN_CLASSIFIER", help='')
    parser.add_argument("ERROR_SPAN_GENERATOR", help='')
    parser.add_argument("REF_M2", help='')
    parser.add_argument("OUT_CSV", help='')
    args = parser.parse_args()
    
    main(args.SENTENCES, args.TYPE_CLASSIFIER, args.SPAN_CLASSIFIER, args.ERROR_SPAN_GENERATOR, args.REF_M2, args.OUT_CSV)