from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np

from six.moves import cPickle as pickle

def save(train_d, train_l, valid_d, valid_l, test_d, test_l):
	np.save('train_dataset', train_d)
	np.save('train_labels', train_l)
	np.save('valid_dataset', valid_d)
	np.save('valid_labels', valid_l)
	np.save('test_dataset', test_d)
	np.save('test_labels', test_l)

def load():
	train_d = np.load('train_dataset.npy')
	train_l = np.load('train_labels.npy')
	valid_d = np.load('valid_dataset.npy')
	valid_l = np.load('valid_labels.npy')
	test_d = np.load('test_dataset.npy')
	test_l = np.load('test_labels.npy')
	print("Train set:", train_d.shape, train_l.shape)
	print("Valid set:", valid_d.shape, valid_l.shape)
	print("Test  set:", test_d.shape, test_l.shape)
	return train_d, train_l, valid_d, valid_l, test_d, test_l