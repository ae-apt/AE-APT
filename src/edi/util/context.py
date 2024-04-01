
import csv

# This class reads csv data that is of the following form:
# header row: A0,A1,...,An
# data row: UUID, V1, ..., Vn
# Attribute A0 is ignored.  A1...An are taken to be the attribute names.
# Each row starts with an object identifier and a series of values.
# The values are assumed to be 0/1 only.
# This data is represented internally as a dictionary mapping UUIDs to
# dictionaries mapping keys (A1...An) to their 0/1 values.


class Context:

	def __init__(self,reader):
		self.header = next(reader)[1:]
		self.data = {row[0]:{h : v
							 for (h,v) in zip(self.header,map(lambda x: int(x),row[1:]))}
						for row in reader}


def getContext(ctxtfile):
    with open(ctxtfile) as infile:
        reader = csv.reader(infile)
        ctxt = Context(reader)
    return ctxt
