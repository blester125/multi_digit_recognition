from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from six.moves import cPickle as pickle

data_set_size = {'train': 0, 'valid': 0, 'test': 0}
sequence_lengths = {'train': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
					'extra': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
					'test' : {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}}
max_height = {'train': 0, 'extra': 0, 'test': 0}
max_width = {'train': 0, 'extra': 0, 'test': 0}
means = {'train': 0., 'test': 0., 'extra': 0.}
stds = {'train': 0., 'test': 0., 'extra': 0.}
train_samples = []
test_samples = []
NUM_LABELS = 10
NUM_LENGTHS = 7

test_accuracy = 0.0
train_accuracy = 0.0
valid_accuracy = 0.0
train_time = 0.0

def save():
	filename = "Analytics.pkl"
	f = open(filename, "wb")
	save = {
		"data_set_size": data_set_size,
		"sequence_lengths": sequence_lengths,
		"max_height": max_height,
		"max_width": max_width,
		"train_samples": train_samples,
		"test_samples": test_samples,
		"means": means,
		"stds": stds,
		"train_time": train_time,
		"train_accuracy": train_accuracy,
		"valid_accuracy": valid_accuracy,
		"test_accuracy": test_accuracy
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()

def load():
	global data_set_size
	global sequence_lengths
	global max_height
	global max_width
	global train_samples
	global test_samples
	global means
	global stds
	global train_time
	global train_accuracy
	global valid_accuracy
	global test_accuracy
	filename = "Analytics.pkl"
	try:
		f = open(filename, "rb")
	except:
		save()
		f = open(filename, "rb")
	saved = pickle.load(f)
	data_set_size = saved["data_set_size"]
	sequence_lengths = saved["sequence_lengths"]
	max_height = saved["max_height"]
	max_width = saved["max_width"]
	train_time = saved["train_time"]
	train_samples = saved["train_samples"]
	test_samples = saved["test_samples"]
	means = saved["means"]
	stds = saved["stds"]
	train_accuracy = saved["train_accuracy"]
	valid_accuracy = saved["valid_accuracy"]
	test_accuracy = saved["test_accuracy"]
	del saved

def display():
	print("\nAnalytics")
	print("Sizes of the datasets:")
	for dataset in data_set_size:
		print(dataset + ": " + str(data_set_size[dataset]))
	print("Number of samples in each length class:")
	for dataset in sequence_lengths:
		print(dataset, ":")
		for length in sequence_lengths[dataset]:
			print(str(length) + ": " + str(sequence_lengths[dataset][length]))
	print("Max Height of an image: " + str(get_max_height()))
	print("Max Width of an image: " + str(get_max_width()))
	print("Max Height of training: " + str(max_height["train"]))
	print("Max Height of extra: " + str(max_height["extra"]))
	print("Max Height of test: " + str(max_height["test"]))
	print("Max width of training: " + str(max_width["train"]))
	print("Max width of extra: " + str(max_width["extra"]))
	print("Max width of test: " + str(max_width["test"]))
	print("Mean pixel values of the images:")
	print("Training:", means['train'])
	print("Extra:", means['extra'])
	print("Test:", means['test'])
	print("Standard Devation of pixel values:")
	print("Training:", stds['train'])
	print("Extra:", stds['extra'])
	print("Test:", stds['test'])
	# from Visualize import display_example
	# for t in train_samples:
	# 	display_example(t, 'train')
	# for t in test_samples:
	# 	display_example(t, 'test')
	print("Train Time:", train_time / 60 / 60, "hours")
	print("Train Accuracy:", train_accuracy)
	print("Valid Accuracy:", valid_accuracy)
	print("Test Accuracy:", test_accuracy)
	print("")

def get_max_height():
	return np.amax(max_height.values())

def get_max_width():
	return np.amax(max_width.values())
