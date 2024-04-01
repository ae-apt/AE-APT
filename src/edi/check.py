import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import edi.util.check as check


parser = argparse.ArgumentParser(description='Checking scores against ground truth')
parser.add_argument('--input',
                    '-i',
                    help='input score file',
					required=True)
parser.add_argument('--output', '-o',
                    help='output score file',
					required=True)
parser.add_argument('--groundtruth', '-g',
                    help='ground truth file',
					required=True)
parser.add_argument('--ty', '-t',
                    help='type of ground truth to use',
                    default=None)
parser.add_argument('--reference', '-f',
                    help='reference scorefile to use (for total number of transactions)',
                    default=None)
parser.add_argument('--reverse','-r',
					help='sort anomaly scores in increasing order',
					action='store_true')
parser.add_argument('--verbose', '-v',
					help="print detailed messages",
					action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    check.main(args.input,args.output,args.groundtruth,
		 args.ty,reference=args.reference,reverse=args.reverse,debug=args.verbose)
