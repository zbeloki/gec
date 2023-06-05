import utils
import m2

import argparse
import pdb

def main(train_m2_fpath, use_simple_types, out_fpath):

    error_types = m2.error_types(simplified=use_simple_types)
    print(f"Working with {len(error_types)} error types (check --simple-types argument)")

    train = utils.m2_to_dataset(train_m2_fpath, use_simple_types=use_simple_types, invert=True)

    def preprocess(examples):
        examples['source'] = utils.to_esg_input_format(
            examples['sentence'],
            examples['error_type'],
            examples['span']
        )
        examples['target'] = examples['target_span']
        return examples
    
    train = train.map(preprocess, batched=False, remove_columns=train.column_names)
    train.to_pandas().to_csv(out_fpath, index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("TRAIN_M2", help='M2 file containing the training data')
    parser.add_argument("OUT_TSV", help='Output file containing the data in the following format: "type SEP correct_span SEP masked_context \t error_span')
    parser.add_argument("--simple-types", action='store_true', help='Use simplified types (24) or original types (54)')
    args = parser.parse_args()
    
    main(args.TRAIN_M2, args.simple_types, args.OUT_TSV)

