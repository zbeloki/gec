import utils

from collections import Counter
import argparse
import pdb

def main(types_fpath, ref_fpath):

    with open(types_fpath, 'r') as f:
        inferred_types = [ ln.strip() for ln in f ]
    inferred_counts = Counter(inferred_types)
    inferred_rates = { t: inferred_counts[t]/len(inferred_types) for t in inferred_counts.keys() }
    
    use_simple_types = inferred_types[0][:2] not in ['R:', 'U:', 'M:']
    ref_rates = utils.load_type_rates(ref_fpath, use_simple_types)
    types = ref_rates.keys()

    for t in types:
        if t not in inferred_rates:
            inferred_rates[t] = 0.0

    mse = sum([ (inferred_rates[t]*100-ref_rates[t]*100)**2 for t in types ]) / len(types)
    print(f"MSE: {mse:.2f}")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("TYPES", help='')
    parser.add_argument("REF_M2", help='')
    args = parser.parse_args()
    
    main(args.TYPES, args.REF_M2)
