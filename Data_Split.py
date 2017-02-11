from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import numpy as np

import Analytics

##
# In this split "classes" of data are the numbers each sequence starts with
def split(data, labels, length):
	random.seed(1337)
	n_labels = 10
	train_index = []
	valid_index = []

	for i in np.arange(n_labels):
		# Add the first 400 index's in training labels that start with i 
		valid_index.extend(np.where(labels[:,1] == i)[0][:length].tolist())
		# Add the rest of the index's to this list
		train_index.extend(np.where(labels[:,1] == i)[0][length:].tolist())

	# Randomize the lists
	random.shuffle(valid_index)
	random.shuffle(train_index)

	# add the extra_data that the valid_index2 index's and the train_data at the 
	# valid_index index's. Then shuffle the data so the labels are the first 
	# column the pixles are the 2, 3rd and the colors (RGB) are the last one.
	valid_data = data[valid_index,:,:,:]
	# Do the same thing with the valid set lables
	valid_labels = labels[valid_index, :]
	# Do the same thing with the training data
	train_data = data[train_index,:,:,:]
	# Do the same thing with the training labels
	train_labels = labels[train_index,:]

	print("Training set created with shape")
	print(train_data.shape, train_labels.shape)
	print("Validation set created with shape")
	print(valid_data.shape, valid_labels.shape)

	#Analytics.load()
	#Analytics.data_set_size['train'] = train_data.shape[0]
	#Analytics.data_set_size['valid'] = valid_data.shape[0]
	#Analytics.save()

	return train_data, train_labels, valid_data, valid_labels