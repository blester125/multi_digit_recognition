from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from skimage import color
from scipy.misc import imresize

import Analytics

def find_bounding_box(digitStruct):
	tops = []
	lefts = []
	lefts_width = []
	tops_height = []
	boxes = digitStruct['boxes']
	for box in boxes:
		tops.append(box['top'])
		lefts.append(box['left'])
		lefts_width.append(box['left'] + box['width'])
		tops_height.append(box['top'] + box['height'])
	top = np.amin(tops)
	left = np.amin(lefts)
	width = np.amax(lefts_width) - left
	height = np.amax(tops_height) - top
	return top, left, width, height

def scale(top, left, width, height):
	new_width = width + (width * 0.3)
	new_height = height + (height * 0.3)
	new_top = top - (height * 0.15)
	new_left = left - (width * 0.15)
	return new_top, new_left, new_width, new_height

def grayscale(img):
	img = np.dot(img, [[0.2989],[0.5870],[0.1140]])
	return img

def resize(img):
	img = imresize(img, (50,50))
	return img

def crop(im, top, left, width, height):
	if top < 0:
		top = 0
	if left < 0:
		left = 0
	crop_im = im[top:top+height, left:(left+width)]
	return crop_im


def normalize(dataset):
	mean = np.mean(dataset)
	std = np.std(dataset)
	if std < 1e-4:
		std = 1
	dataset -= mean
	dataset /= std
	return dataset

def preprocess_image(digitStruct, dataset=""):
	filename = os.path.join(dataset, digitStruct['filename'])
	im = mpimg.imread(filename)

	# Analytics code
	Analytics.load()
	if im.shape[0] > Analytics.max_height[dataset]:
		Analytics.max_height[dataset] = im.shape[0]
	if im.shape[1] > Analytics.max_width[dataset]:
		Analytics.max_width[dataset] = im.shape[1]
	Analytics.save()

	t, l, w, h = find_bounding_box(digitStruct)
	t, l, w, h = scale(t, l, w, h)
	cropped = crop(im, t, l, w, h)
	resized = resize(cropped)
	gray = grayscale(resized)
	return gray

def preprocess_camera(image, t, l, w, h):
	cropped = crop(image, t, l, w, h)
	resized = resize(cropped)
	gray = grayscale(resized)
	return gray

def preprocess_file_image(image):
	resized = resize(image)
	gray = grayscale(resized)
	return gray