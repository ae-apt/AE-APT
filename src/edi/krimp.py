import argparse
import csv
import glob
import numpy as np
import os
import random
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import edi.util.context as context

parser = argparse.ArgumentParser(description='Krimp/OC3')
parser.add_argument('--input', '-i',
					help='input file', required=True)
parser.add_argument('--output', '-o',
					help='output file', required=True)
parser.add_argument('--codetable', '-c',
					help='code table file')


def getPair(s):
    m = re.match(r"\((\d+),(\d+)\)", s)
    (uses,supp) = m.groups()
    return (int(uses),int(supp))

def getIdMap(datafile,ctxt):
    with open(datafile) as f:
        lines = f.readlines()
        ids = lines[2].split()[1:]
        oldids = lines[5].split()[1:]
    return {int(x):ctxt.header[int(y)-1] for (x,y) in zip(ids,oldids)}

def getCodeTable(filename,idmap):
    with open(filename) as f:
        lines = f.readlines()
        header = lines[0].split()
        sizes = lines[1].split()
        rest = lines[2:]
    splitlines = [line.split() for line in rest]
    n = sum([getPair(line[-1])[0]+1 for line in splitlines])
    ct = [{'attributes': {idmap[int(x)] for x in line[0:-1]},
           'cost': -np.log2((getPair(line[-1])[0]+1)/n)} for line in splitlines]
    return ct

def writeCodeTable(f,ctxt,ct):
    f.write('cost,%s\n' % ','.join(ctxt.header))
    for cte in ct:
        values = ','.join(['1' if att in cte['attributes'] else '0' 
                           for att in ctxt.header])
        f.write('%f,%s\n' % (cte['cost'], values))

def merge(input,context,output):
	with open(input) as score_file:
		score_reader = csv.reader(score_file)
		header = next(score_reader)[1:]
		scores = [float(row[1])
					for row in score_reader]

	with open(context) as context_file, open(output,'w') as out_file:
		context_reader = csv.reader(context_file)
		idx = 0
		header = next(context_reader)[1:]
		out_file.write("Object_ID, OC3 Score\n")
		for row in context_reader:
			out_file.write("%s,%f\n" % (row[0],scores[idx]))
			idx = idx+1

def run(input,output,codetable=None):
	prefix = '/tmp/krimp%s' % random.randint(1,10000000)
	krimpbin = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bin/krimp'))
	mergepy = os.path.abspath(os.path.join(os.path.dirname(__file__), 'merge.py'))

	dat_file = prefix+'/data/datasets/data.dat'
	db_file = prefix+'/data/datasets/data.db'

	# todo: create a uniquely named directory so that we can run
	# multiple jobs concurrently without interference
	for d in [prefix,prefix+'/data',prefix+'/data/datasets']:
		os.mkdir(d)

	# inlined from convert.py
	with open(input) as csv_file, open(dat_file,'w') as out_file:
		reader = csv.reader(csv_file)
		header = next(reader)[1:]
		for row in reader:
			items = [str(i+1) for i,x in enumerate(row[1:]) if int(x) == 1]
			out_file.write("%s\n" % (" ".join(items)))
		allitems = [str(i+1) for i in range(len(header))]
		out_file.write("%s\n" % (" ".join(allitems)))

	with open(prefix+'/convert.conf','w') as convert_file:
		convert_file.write("""taskclass = datatrans
command = convertdb
takeItEasy = 0
dataDir = %s/data/
expDir = %s/xps/
dbName = data
dbInEncoding = fimi
dbInExt = dat
dbOutExt = db
dbOutEncoding = fic
dbOutTranslateFw = true
dbOutOrderInTrans = true
dbOutBinned = false""" % (prefix,prefix))

	with open('%s/compress.conf' % prefix,'w') as convert_file:
		convert_file.write("""algo = coverpartial
command = compress
datadir = %s/data/
expdir = %s/xps/
internalmineto = memory
iscchunktype = bisc
iscifmined = zap
iscname = data-closed-1d
iscstoretype = bisc
maxmemusage = 1024
numthreads = 1
prunestrategy = pop
reportacc = false
reportcnd = 0
reportsup = 100
takeiteasy = 0
taskclass = main""" % (prefix,prefix))

	os.system('%s %s/convert.conf > %s/convert.log' % (krimpbin,prefix,prefix))
	os.system('%s %s/compress.conf > %s/compress.log' % (krimpbin,prefix,prefix))

	#slight hack, should find the generated file and do merge directly
	os.system('cp %s/xps/compress/*/el*.csv %s/scores.csv' % (prefix,prefix))
	merge(prefix+'/scores.csv',input,output)

	if codetable != None:
		ctxt = context.getContext(input)
		idmap = getIdMap(db_file,ctxt)
		ctfile = glob.glob('%s/xps/compress/*/ct*-1-*.ct' % prefix)
		ct = getCodeTable(ctfile[0],idmap)
		with open(codetable,'w') as f:
			writeCodeTable(f,ctxt,ct)
	shutil.rmtree(prefix)

if __name__ == '__main__':
	args = parser.parse_args()
	start_time = time.time()
	run(args.input,args.output,args.codetable)
	print("krimp completed: %s seconds ---" % (time.time() - start_time))
