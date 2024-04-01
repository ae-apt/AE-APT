import argparse
import csv
import gzip
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import edi.simple.ad as ad

parser = argparse.ArgumentParser(description='Attribute value frequency')
parser.add_argument('--input', '-i',
					help='input file', required=True)
parser.add_argument('--output', '-o',
					help='output file', required=True)
parser.add_argument('--score', '-s',
					help='avf scoring',
					default='avf')
parser.add_argument('--mode', '-m',
					help='batch or stream',
					default='batch',
					choices=['batch','stream'])


if __name__ == '__main__':
	args = parser.parse_args()

	if(args.mode == 'batch'):
		start=time.time()
		ad.batch(args.input, args.output, args.score)
		end=time.time()
		print(end - start)
elif (args.mode == 'stream'):
		start=time.time()
		ad.stream(args.input, args.output, args.score)
		end=time.time()
		print(end - start)
