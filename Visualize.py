from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw

import Preprocess

##
# Display an image with digit bounding boxes. The digitStruct should be from
# one of the data sets. The dataset parameter should be the path to the folder 
# with the images in them.
def display_example(digitStruct, dataset=""):
	filename = os.path.join(dataset, digitStruct['filename'])
	im = mpimg.imread(filename)
	boxes = digitStruct['boxes']
	for box in boxes:
		#print box
		top = box['top']
		left = box['left']
		width = box['width']
		height = box['height']
		draw_box(plt, top, left, width, height)
	imgplot = plt.imshow(im, cmap=plt.cm.binary)
	# Code to the draw the large bounding box
	#t, l, w, h = Preprocess.find_bounding_box(digitStruct)
	#draw_box(plt, t, l, w, h)
	#t, l, w, h = Preprocess.scale(t, l, w, h)
	#draw_box(plt, t, l, w, h)
	plt.show()

def load_example(digitStruct, dataset=""):
	filename = os.path.join(dataset, digitStruct['filename'])
	im = mpimg.imread(filename)
	return im

def display_processed_example(img):
	disp = img[:, :, 0]
	imgplot = plt.imshow(disp, cmap=plt.cm.binary)
	plt.show()

##
# Draw a bounding box from the top left corner and the width/height
def draw_box(plt, top, left, width, height):
	# In a plot the x values are horizonatal, y's are vertical

	# Plot from (left, top) to (left + width, top)
	# This is the top line
	plt.plot(
		[left, left + width], 
		[top, top], 
		color='k', 
		linestyle='-', 
		linewidth=2
	)
	# Plot from (left, top + height) to (left, top)
	# This is the left hand line
	plt.plot(
		[left, left], 
		[top + height, top], 
		color='k', 
		linestyle='-', 
		linewidth=2
	)
	# Plot from (left, top+height) to (left + width, top+height)
	# This is the bottom line
	plt.plot(
		[left, left + width], 
		[top + height, top + height], 
		color='k', 
		linestyle='-', 
		linewidth=2
	)
	# Plot from (l+w, t+h) to (l+w, t)
	# This is the right hand line
	plt.plot(
		[left + width, left + width], 
		[top + height, top], 
		color='k', 
		linestyle='-', 
		linewidth=2
	)
