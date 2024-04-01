import numpy

# A model is a structure with two operations.
# "score" which takes a transaction and scores it.
# "update" which takes a transaction and updates the model.

# The base class provides generic implememtations of these operations.
# Score returns 1 while update maintains counts of the frequencies of attributes seen
# so far.
# Additional operations analyze and adapt are derived and shouldn't be overridden.
# analyze takes a list and updates all the elements of the list, then returns
# a list of scores using the final model.
# adapt takes a list and alternates scoring and updating, so that each item
# is scored using the current version fo the model based on the data seen so far.
# TODO: Make a nicer streaming version of adapt.
# TODO: Currently scores are recomputed from-scratch using numpy.log2,
# probably resulting in many redundant calculations, especially in batch mode.
# Adding some caching could speed things up considerably.


class Model:

	def __init__(self,attrs=set()):
		self.attrs = attrs
		self.freqs = {attr:0 for attr in attrs}
		self.n = 0

	def score(self,x):
		return 1.0

	def update(self,x):
		for att in x.keys():
			if att in self.attrs:
				self.freqs[att] = self.freqs[att] + x[att]
			else:
				self.freqs[att] = x[att]
				if x[att] > 0:
					self.attrs.add(att)
		self.n = self.n+1

	def analyze(self,l):
		for x in l:
			self.update(x)
		return [self.score(x) for x in l]

	def adapt(self,l): 
		def handle(x):
			s = self.score(x)
			self.update(x)
			return s
		return [handle(x) for x in l]


# AVF models use the frequencies to estimate attribute probabilities, and the
# score of an item is the sum of the probabilities of its attributes.
# This is not a MDL-based or particularly principled technique but
# is fast and useful.
class AVFModel(Model):

	def score(self,x):
		score = 0
		for att in x.keys():
			p = self.freqs[att] / self.n
			score = score + x[att] * p + (1-x[att]) * (1-p)
		return score

# For the online version, we use Laplace correction to ensure the denominator
# is nonzero at the beginning.
class AVFOnlineModel(Model):

	def score(self,x):
		score = 0
		for att in x.keys():
			p = (self.freqs[att] + 1) / (self.n + 2)
			score = score + x[att] * p + (1-x[att]) * (1-p)
		return score

# AVC is similar to AVF but the overall scores are codelengths based on the
# estimated probabilities.  Note that we Laplace correct in both the streaming
# and batch cases.  This is because
class AVCModel(Model):

	def score(self,x):
		score = 0.0
		for att in x.keys():
			p = (self.freqs[att] + 1) / (self.n + 2)
			score = score - x[att] * numpy.log2(p) - (1-x[att]) * numpy.log2(1-p)
		return score


class AVCOnlineModel(Model):

	def score(self,x):
		score = 0.0
		for att in x.keys():
			p = (self.freqs[att] + 1) / (self.n + 2)
			score = score - x[att] * numpy.log2(p) - (1-x[att]) * numpy.log2(1-p)
		return score


# Rasp is an experimental approach based on compressing only the '1' attributes
# and using special control characters to signal 'end of transaction'
# It is inspired by / simpler than Krimp, another MDL-based technique
# In an offline setting it provides similar results to AVC.
# One potential advantage (in the online case) is that it can deal with
# the attribute set not being known at the beginning.
class RaspModel(Model):

	def score(self,x):
		score = 0.0
		total = self.n
		for att in self.attrs:
			total = total + self.freqs[att]
		for att in x.keys():
			p = self.freqs[att] / total
			score = score - x[att] * numpy.log2(p)
		score = score - numpy.log2(self.n / total)
		return score

# For the online version, we again want to avoid dividing by zero
#	at the beginning, and also we need another control symbol to deal with
# any new attributes encountered during stream processing.  At the beginning
# we assume we have seen no previous attributes.

class RaspOnlineModel(Model):

	def score(self,x):
		score = 0.0
		total = self.n + 1 + 1
		for att in self.attrs:
			total = total + (self.freqs[att])
		for att in x.keys():
			if att in self.attrs:
				p = (self.freqs[att]+1) / (total+2)
				score = score - x[att] * numpy.log2(p)
			else:
				p = 1/total
				score = score - x[att] * numpy.log2(p)

		score = score - numpy.log2((self.n + 1) / total)
		return score

# Both RASP models are inefficient compressors in the sense that they don't
# take advantage of the fact that the transactions are sets
# (hence order and multiplicity don't matter).  However this doesn't appear to
# make a big difference to the anomaly detection results.
