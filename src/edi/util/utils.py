import csv
import gzip


def readRecord(header,row):
	uuid = row[0]
	record = {h : v for (h,v) in zip(header,map(lambda x: int(x),row[1:]))}
	return (uuid,record)


class BatchProcessor:
	def __init__(self,input_file, output_file, model_name, mk_model):
		self.input_file = input_file
		self.output_file = output_file
		self.model_name = model_name
		self.mk_model = mk_model

	def create_model(self,csv_file):
		reader = csv.reader(csv_file)
		header = next(reader)[1:]

		print('Scoring batch using %s' % self.model_name)
		score_header = 'Object_ID,%sScore\n' % self.model_name
		m = self.mk_model(header)

		for row in reader:
			(uuid,record) = readRecord(header,row)
			m.update(record)
		return (m,score_header)


	def write_scores(self,csv_file,score_file,m,score_header):
		reader = csv.reader(csv_file)
		header = next(reader)[1:]
		score_file.write(score_header)

		for row in reader:
			(uuid,record) = readRecord(header,row)
			score = m.score(record)
			score_file.write("%s, %f\n" % (uuid,score))

	def process(self):
		if self.input_file.endswith('.gz'):
			with gzip.open(self.input_file,'rt') as csv_file:
				(m,score_header) = self.create_model(csv_file)
			with gzip.open(self.input_file,'rt') as csv_file, open(self.output_file,'w') as score_file:
				self.write_scores(csv_file,score_file,m,score_header)
		else:
			with open(self.input_file,'rt') as csv_file:
				(m,score_header) = self.create_model(csv_file)
			with open(self.input_file,'rt') as csv_file, open(self.output_file,'w') as score_file:
				self.write_scores(csv_file,score_file,m,score_header)



class StreamProcessor:
	def __init__(self,input_file, output_file, model_name, mk_model):
		self.input_file = input_file
		self.output_file = output_file
		self.model_name = model_name
		self.mk_model = mk_model

	def handle_stream(self,csv_file,score_file):
		reader = csv.reader(csv_file)
		header = next(reader)[1:]

		print('Scoring stream using %s' % self.model_name)
		score_header = 'Object_ID,%sScore\n' % self.model_name
		m = self.mk_model(header)

		score_file.write(score_header)
		totalscore = 0.0

		for row in reader:
			(uuid,record) = readRecord(header,row)
			score = m.score(record)
			totalscore = totalscore + score
			m.update(record)
			score_file.write("%s, %f\n" % (uuid,score))
		print("Total score: %f  Entropy: %f" % (totalscore, totalscore/m.n))

	def process(self):
		if self.input_file.endswith('.gz'):
			with gzip.open(self.input_file,'rt') as csv_file, open(self.output_file,'w') as score_file:
				self.handle_stream(csv_file,score_file)
		else:
			with open(self.input_file,'rt') as csv_file, open(self.output_file,'w') as score_file:
				self.handle_stream(csv_file,score_file)


