import argparse
import pandas as pd
import numpy as np
import pyfpgrowth
import csv
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os.path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter


parser = argparse.ArgumentParser(description='Attribute value frequency')
parser.add_argument('--input', '-i',
					help='input file', required=True)
parser.add_argument('--output', '-o',
					help='output file', required=True)
parser.add_argument('--score', '-s',
					help='fpof or od',
					default='fpof',
					choices=['fpof','od'])
parser.add_argument('--conf','-c',
					help='Minimum confidence',
					type=float,
					default=0.9)
parser.add_argument('--minsupp','-m',
					help='Minimum support',
					type=float,
					default=0.2)

args = parser.parse_args()


infile = args.input

outfile = args.output

context	= pd.read_csv(infile)
modellingdata = context.drop('Object_ID', axis=1)


# Scoring Algorithms - Experiments across the min_sup & min_conf thresholds

# Create the list for the pfgrowth algo - substitute the 1s in each column with the resp. column name
for ncol in range(modellingdata.shape[1]):
	tmd = modellingdata.iloc[:, [ncol]].astype(str)
	cnm = tmd.columns.values
	if ncol == 0:
		modellingdata_1 = tmd.replace(['1'], cnm)
	if ncol > 0:
		tmd = tmd.replace(['1'], cnm)
		modellingdata_1 = pd.concat([modellingdata_1, tmd], axis=1)
		#Create the list for the pfgrowth algo - create the list
df = modellingdata_1.loc[(modellingdata_1!="0").any(axis=1)] # Drop rows with all zeros
modelinput = df.values.tolist() # pfgrowth needs a list input
# Clean the list of the zeros
def tuple_without(original_tuple, element_to_remove):
	new_tuple = []
	for s in list(original_tuple):
		if not s == element_to_remove:
			new_tuple.append(s)
	return tuple(new_tuple)
txn_list = []
for i, mytuple in enumerate(modelinput):
	txn_list.append(tuple_without(mytuple, "0"))



# Function definition: PFGrowth algorithm - fetching the Patterns (Frequent Itemsets) for FPOF and/or the Rules for OD
def get_patterns_rules(txn_list, 
					   FPOF_flag=0, 
					   OutliernessDegree_flag=0, 
					   min_sup=np.nan, 
					   min_conf=np.nan):
	print("FPGrowth rule-mining in progress ......")
	# Run PFGRWOTH ALGO - Get the Frequent patterns
	minsup = min_sup
	minconf = min_conf
	print("Min Support: ", minsup, ", Min Confidence: ", minconf)
	leninput = len(modelinput)
	if FPOF_flag == 1:
		patterns = pyfpgrowth.find_frequent_patterns(txn_list, minsup*leninput)
		print("Number of patterns (Frequent Itemsets):  ", len(patterns))
		print("FPGrowth rule-mining completed")
		return patterns
	if OutliernessDegree_flag == 1:
		patterns = pyfpgrowth.find_frequent_patterns(txn_list, minsup*leninput)
		rules = pyfpgrowth.generate_association_rules(patterns, minconf)
		print("Number of patterns (Frequent Itemsets):  ", len(patterns), ", Number of high-confidence rules:  ", len(rules))
		print("FPGrowth rule-mining completed")
		return rules	

# Function defintion - FPOF scoring technique
def compute_fpof_scores(patterns,modellingdata_0a):
	print("FPOF Scoring in progress ....")
	# Calculate FPOF score - Convert the patterns dictionary to a list
	frequentitemsets = []
	ntxn = []
	for key, value in patterns.items():
		frequentitemsets.append(key)
		ntxn.append(value)
	# Calculate FPOF score
	txn_array = np.asarray(txn_list)
	txn_array = np.reshape(txn_array, (txn_array.size, 1))
	frequentitemsets_array = np.asarray(frequentitemsets)
	frequentitemsets_array = np.reshape(frequentitemsets_array, (frequentitemsets_array.size, 1))
	total_ntxn = len(txn_list)
	s_txn = 0
	e_txn = total_ntxn + 1
	txncounter = s_txn
	if s_txn == 0:
		modellingdata_0a['FPOF_Score'] = 0
	start_time = time.time()
	for txn in txn_array[s_txn:e_txn]:
		tarr = np.array([z for z in txn])
		ficounter = 0
		fpof_score = 0
		#print("Completed TXN number: ", txncounter)
		for freqitem in frequentitemsets_array:
			fiarr = np.array([z for z in freqitem])
			if tarr.size > fiarr.size: # Avoids wasting of time
				containsarray = np.in1d(tarr, fiarr)
				ntrue = np.sum(containsarray)
				lfiarr = fiarr.size
				if ntrue == lfiarr:
					fpof_score += ntxn[ficounter]/total_ntxn
			ficounter = ficounter + 1
		modellingdata_0a.iloc[txncounter, modellingdata_0a.columns.get_loc('FPOF_Score')] = fpof_score
		txncounter = txncounter + 1
	print("FPOF scoring completed: %s seconds ---" % (time.time() - start_time))
	return modellingdata_0a

# Function defintion - Outlierness degree scoring technique
def compute_od_scores(rules,modellingdata_0a):
	print("Outlierness Degree Scoring in progress ....")
	#Remove rules with 100% confidence
	rules_w_conf_lt1 = {k: v for k, v in rules.items() if v[1] < 1}
	#Convert the rules dictionary to lists of LHS & RHS
	lhs = []
	rhs0 = []
	for key, value in rules_w_conf_lt1.items():
		lhs.append(key)
		rhs0.append(value)
	# Removethe confidence values from RHS
	rhs = [item[0] for item in rhs0]
	# Compute the Outlierness Degree w.r.t. the High-confidence rules
	# Convert all lists to numpy arrays for efficiency
	txn_array = np.asarray(txn_list)
	txn_array = np.reshape(txn_array, (txn_array.size, 1))
	lhs_array = np.asarray(lhs)
	lhs_array = np.reshape(lhs_array, (lhs_array.size, 1))
	rhs_array = np.asarray(rhs)
	rhs_array = np.reshape(rhs_array, (rhs_array.size, 1))
	total_ntxn = len(txn_list)
	start_time = time.time()
	s_txn = 0
	e_txn = total_ntxn + 1
	txncounter = s_txn
	if s_txn == 0:
		modellingdata_0a['OutlierDegree_Score'] = 0
	for txn in txn_array[s_txn:e_txn]:
		tarr = np.array([z for z in txn])
		ruleindex = 0
		associativeclosure_arr = tarr.flatten()
		for rhsitem in rhs_array:
			rhsarr = np.array([z for z in rhsitem])
			if tarr.size > rhsarr.size: # Avoids wasting of time
				containsarray = np.in1d(tarr, rhsarr)
				ntrue = np.sum(containsarray)
				lrhsarr = rhsarr.size
				if ntrue == lrhsarr:
					lhsarr = np.array([z for z in lhs_array[ruleindex]])
					associativeclosure_arr = np.union1d(associativeclosure_arr.flatten(), lhsarr.flatten())
			ruleindex = ruleindex + 1
		outlierdegree = (associativeclosure_arr.size - tarr.flatten().size)/associativeclosure_arr.size
		modellingdata_0a.iloc[txncounter, modellingdata_0a.columns.get_loc('OutlierDegree_Score')] = outlierdegree
		txncounter = txncounter + 1
	print("Outlierness degree scoring completed: %s seconds ---" % (time.time() - start_time))
	return modellingdata_0a


def do_fpof_scoring(min_sup):
	patterns = get_patterns_rules(txn_list,
								  FPOF_flag=1,
								  min_sup=min_sup)
	# Generate the Transactional modelling data fresh for every experiment
	modellingdata_0a = context.filter(['Object_ID'], axis=1)
	##################
	fpofscores = compute_fpof_scores(patterns,modellingdata_0a)
	fpofscores.to_csv(outfile,index=False)

def do_od_scoring(min_sup,min_conf):
	rules = get_patterns_rules(txn_list,
							   OutliernessDegree_flag=1,
							   min_sup=min_sup,
							   min_conf=min_conf)
	# Generate the Transactional modelling data fresh for every experiment
	modellingdata_0a = context.filter(['Object_ID'], axis=1)
	##################
	odscores = compute_od_scores(rules,modellingdata_0a)
	odscores.to_csv(outfile,index=False)


if args.score == 'fpof':
		start=time.time()
		do_fpof_scoring(args.minsupp)
		end=time.time()
		with open("output.txt",'a') as f:
			print(infile, file=f, end='\n')	
			tmp=end-start
			print(tmp, file=f, end='\n')			
        

elif args.score == 'od':
	start=time.time()
	do_od_scoring(args.minsupp,args.conf)
	end=time.time()
	print(end - start)
 
