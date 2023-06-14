from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import torch

from tokenizer import Tokenizer
import argparse

import os
import pdb

MAX_LENGTH = 256
DEF_BATCH_SIZE = 32

device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

def generate_predictions(source_sentences, pipe, batch_size):

    predictions = pipe(source_sentences, batch_size=batch_size)
    pred_sents = [ p['generated_text'] for p in predictions ]
    return pred_sents


def predict(model_path, in_fpath, out_fpath, tok_engine, batch_size):

    tokenizer = Tokenizer(engine=tok_engine)
    bart_tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    pipe = pipeline('text2text-generation', max_length=MAX_LENGTH, tokenizer=bart_tokenizer, model=model, device=device)

    with open(in_fpath, 'r') as f:
        src_sents = [ ln.strip() for ln in f.readlines() ]
    src_sents = [ tokenizer.detokenize(sent) for sent in src_sents ]
    pred_sents = generate_predictions(src_sents, pipe, batch_size)
    #pred_sents = generate_predictions(pred_sents, pipe, batch_size) # ERNE!!! Proba bat da!!!
    pred_sents_toked = [ ' '.join(tokenizer.tokenize(s)) for s in pred_sents ]

    with open(out_fpath, "w") as f:
        for pred in pred_sents_toked:
            print(pred, file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("MODEL", help='Path of the model to evaluate')
    parser.add_argument("FILE_TXT", help='Text file containing tokenized sentences, line by line')
    parser.add_argument("OUT", help='Output file to write predicted sentences')
    parser.add_argument("--tokenizer", "-t", choices=['spacy', 'nltk'], default='spacy', help='Use Spacy or NLTK to tokenize the resulting sentence')
    parser.add_argument("--batch-size", "-b", type=int, default=DEF_BATCH_SIZE, help='Batch size')
    args = parser.parse_args()

    predict(args.MODEL, args.FILE_TXT, args.OUT, args.tokenizer, args.batch_size)
