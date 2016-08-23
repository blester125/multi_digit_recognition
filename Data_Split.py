from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import numpy as np

import Analytics

##
# In this split "classes" of data are the numbers each sequence starts with
def split(train_data, train_labels, extra_data, extra_labels):
	random.seed()
	n_labels = 10
	valid_index = []
	valid_index2 = []
	train_index = []
	train_index2 = []

	for i in np.arange(n_labels):
		# Add the first 400 index's in training labels that start with i 
		valid_index.extend(np.where(train_labels[:,1] == i)[0][:400].tolist())
		# Add the rest of the index's to this list
		train_index.extend(np.where(train_labels[:,1] == i)[0][400:].tolist())
		# The first 200 from the extra set
		valid_index2.extend(np.where(extra_labels[:,1] == i)[0][:200].tolist())
		# The rest of the extra set
		train_index2.extend(np.where(extra_labels[:,1] == i)[0][200:].tolist())

	# Randomize the lists
	random.shuffle(valid_index)
	random.shuffle(train_index)
	random.shuffle(valid_index2)
	random.shuffle(train_index2)

	# add the extra_data that the valid_index2 index's and the train_data at the 
	# valid_index index's. Then shuffle the data so the labels are the first 
	# column the pixles are the 2, 3rd and the colors (RGB) are the last one.
	valid_data = np.concatenate(
					(extra_data[valid_index2,:,:,:],
					 train_data[valid_index,:,:,:]),
					 axis=0)
	# Do the same thing with the valid set lables
	valid_labels = np.concatenate(
						(extra_labels[valid_index2, :],
						 train_labels[valid_index, :]),
						 axis=0)
	# Do the same thing with the training data
	train_data = np.concatenate(
						(extra_data[train_index2,:,:,:], 
						 train_data[train_index,:,:,:]), 
						 axis=0)
	# Do the same thing with the training labels
	train_labels = np.concatenate(
						(extra_labels[train_index2,:], 
						 train_labels[train_index,:]), 
						 axis=0)

	print("Training set created with shape")
	print(train_data.shape, train_labels.shape)
	print("Validation set created with shape")
	print(valid_data.shape, valid_labels.shape)

	Analytics.load()
	Analytics.data_set_size['train'] = train_data.shape[0]
	Analytics.data_set_size['valid'] = valid_data.shape[0]
	Analytics.save()

	return train_data, train_labels, valid_data, valid_labels