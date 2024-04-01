
from . import model
from ..util import utils


def batch(input, output, score):
	if (score == 'avf'):
		model_name = 'AVF'
		mk_model = model.AVFModel
	elif (score == 'avc'):
		model_name = 'AVC'
		mk_model = model.AVCModel
	elif (score == 'rasp'):
		model_name = 'RASP'
		mk_model = model.RaspModel
	processor = utils.BatchProcessor(input, output,
									 model_name, mk_model)
	processor.process()




def stream(input, output, score):
	if (score == 'avf'):
		model_name = 'AVF'
		mk_model = model.AVFOnlineModel
	elif (score == 'avc'):
		model_name = 'AVC'
		mk_model = model.AVCOnlineModel
	elif (score == 'rasp'):
		model_name = 'RASP'
		mk_model = model.RaspOnlineModel
	processor = utils.StreamProcessor(input, output,
									  model_name, mk_model)
	processor.process()
