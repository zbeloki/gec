from tokenizer import Tokenizer

import tqdm

import argparse
import csv
import pdb

def main(csv_fpath, orig_fpath, cor_fpath):

    tokenizer = Tokenizer(engine='spacy')
    
    sources = []
    targets = []
    with open(csv_fpath, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm.tqdm(reader):
            source = ' '.join(tokenizer.tokenize(row['source']))
            target = ' '.join(tokenizer.tokenize(row['target']))
            sources.append(source)
            targets.append(target)

    with open(orig_fpath, 'w') as f:
        for sent in sources:
            print(sent, file=f)

    with open(cor_fpath, 'w') as f:
        for sent in targets:
            print(sent, file=f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("CSV", help='')
    parser.add_argument("OUT_ORIG", help='')
    parser.add_argument("OUT_COR", help='')
    args = parser.parse_args()

    main(args.CSV, args.OUT_ORIG, args.OUT_COR)
