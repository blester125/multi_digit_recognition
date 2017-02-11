from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from Preprocess import preprocess_image, normalize

import Analytics
import Visualize

def generate_dataset(data, folder, single=False):
	target_size = 64
	if folder == 'test' and not example:
		Analytics.load()
		Analytics.data_set_size[folder] = len(data)
		Analytics.save()
	data_point_per_image = 5
	if single == True:
		data_point_per_image = 1
	dataset = np.ndarray([len(data) * data_point_per_image, target_size, target_size, 1], dtype='float32')
	labels = np.ones([len(data) * data_point_per_image, 6], dtype=int) * 10
	offset = 0
	for i in np.arange(len(data)):
		processed_images = preprocess_image(data[i], folder, single)
		dataset[offset:offset+len(processed_images), :, :, :] = processed_images
		labels[offset:offset+len(processed_images), :] = getLabels(data[i], folder)
		offset += len(processed_images)

	print(folder)
	print(np.mean(dataset))
	print(np.std(dataset))

	# Analytics
	#Analytics.load()
	#Analytics.means[folder] = np.mean(dataset)
	#Analytics.stds[folder] = np.std(dataset)
	#Analytics.save()

	#dataset = normalize(dataset)

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