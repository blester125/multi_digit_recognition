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

def resize(img, size=64):
	img = imresize(img, (size, size))
	return img

def crop(im, top, left, width, height):
	top = int(top)
	left = int(left)
	width = int(width)
	height = int(height)
	if top < 0:
		top = 0
	if left < 0:
		left = 0
	crop_im = im[top:int(top+height), left:int(left+width)]
	return crop_im


def normalize(dataset, in_mean=None, in_std=None):
	mean = np.mean(dataset)
	std = np.std(dataset)
	if std < 1e-4:
		std = 1
	if in_mean is None:
		dataset -= mean
	else:
		dataset -= in_mean
	if in_std is None:
		dataset /= std
	else:
		dataset /= in_std
	return dataset

def preprocess_image(digitStruct, dataset="", single=False):
	processed_images = []

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
	t30, l30, w30, h30 = scale(t, l, w, h)
	w_prime = w + ((w30 - w) / 2)
	h_prime = h + ((h30 - h) / 2)

	cropped = crop(im, t30, l30, w30, h30)
	processed_images.append(cropped)

	if single == False:
		cropped = crop(im, t30, l30, w_prime, h_prime)
		processed_images.append(cropped)

		cropped = crop(im, t30, l, w_prime, h_prime)
		processed_images.append(cropped)

		cropped = crop(im, t, l30, w_prime, h_prime)
		processed_images.append(cropped)

		cropped = crop(im, t, l, w_prime, h_prime)
		processed_images.append(cropped)

	for i in range(len(processed_images)):
		processed_images[i] = resize(processed_images[i])
		processed_images[i] = grayscale(processed_images[i])
		#Visualize.display_processed_example(processed_images[i])

	return processed_images

def preprocess_camera(image, t, l, w, h):
	cropped = crop(image, t, l, w, h)
	resized = resize(cropped)
	gray = grayscale(resized)
	return gray

def preprocess_file_image(image):
	resized = resize(image)
	gray = grayscale(resized)
	return gray