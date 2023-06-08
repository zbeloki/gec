import m2
import tqdm

import argparse
import csv

def convert(m2_fpath, csv_fpath, only_incorrect):

    with open(csv_fpath, 'w') as f_out:
        csv_writer = csv.DictWriter(f_out, fieldnames=['source', 'target'])
        csv_writer.writeheader()
        with open(m2_fpath, 'r') as f_in:
            for m2entry in tqdm.tqdm(m2.read_m2(f_in)):
                if only_incorrect and m2entry.is_correct():
                    continue
                original = m2entry.original()
                aids = m2entry.annotators()
                if len(aids) == 0:
                    csv_writer.writerow({'source': original, 'target': original})
                else:    
                    for aid in aids:
                        correct = m2entry.corrected(annotator=aid)[0]
                        csv_writer.writerow({'source': original, 'target': correct})
    

if __name__ == '__main__':

     parser = argparse.ArgumentParser(description="")
     parser.add_argument("FILE", help='Input M2 file to be converted')
     parser.add_argument("OUT", help='Output CSV file')
     parser.add_argument("--only-incorrect", default=False, action='store_true', help='Output CSV file')
     args = parser.parse_args()
     
     convert(args.FILE, args.OUT, args.only_incorrect)
