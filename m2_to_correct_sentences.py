import m2

import argparse
import pdb

def main(m2_fpath, out_fpath):

    with open(m2_fpath, 'r') as f_in, \
         open(out_fpath, 'w') as f_out:
        for m2_entry in m2.read_m2(f_in):
            correct_sents = m2_entry.corrected()
            if len(correct_sents) == 0:
                pdb.set_trace()
            for sent in set(correct_sents):
                print(sent, file=f_out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("M2", help="M2 file's path")
    parser.add_argument("OUT", help="Out file's path")
    args = parser.parse_args()

    main(args.M2, args.OUT)
