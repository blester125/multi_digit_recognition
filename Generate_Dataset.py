from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from Preprocess import preprocess_image, normalize

import Analytics
import Visualize

def generate_dataset(data, folder):
	if folder == 'test':
		Analytics.load()
		Analytics.data_set_size[folder] = len(data)
		Analytics.save()
	dataset = np.ndarray([len(data), 50, 50, 1], dtype='float32')
	labels = np.ones([len(data), 6], dtype=int) * 10
	for i in np.arange(len(data)):
		dataset[i, :, :, :] = preprocess_image(data[i], folder)
		labels[i, :] = getLabels(data[i], folder)

	# Analytics
	Analytics.load()
	Analytics.means[folder] = np.mean(dataset)
	Analytics.stds[folder] = np.std(dataset)
	Analytics.save()

	dataset = normalize(dataset)

	print(folder, "dataset created.")
	print(dataset.shape)
	print(labels.shape)
	return dataset, labels

def getLabels(digitStruct, folder):
	label = np.ones([6], dtype=int) * 10
	boxes = digitStruct['boxes']
	num_digits = len(boxes)
	label[0] = num_digits
	
	#Analytics Code
	Analytics.load()
	slot = num_digits
	if num_digits > 5:
		slot = 6
	Analytics.sequence_lengths[folder][slot] += 1
	Analytics.save()

	for i in np.arange(num_digits):
		if i < 5:
			label[i + 1] = boxes[i]['label']
			if boxes[i]['label'] == 10:
				label[i + 1] = 0
	return label