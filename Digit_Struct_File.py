from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import range

import h5py
import numpy as np

# Class for accessing h5py data
#  file: The input file (matlab)
#  digitStructName: The ref to all the file names
#  digitStructBbox: The ref to all bounding box data
class DigitStructFile:
	def __init__(self, file):
		self.file = h5py.File(file, 'r')
		self.digitStructName = self.file['digitStruct']['name']
		self.digitStructBbox = self.file['digitStruct']['bbox']

# Returns the 'name' string for for the n digitStruct. 
	def get_name(self,n):
		return ''.join([chr(c[0]) for c in self.file[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
	def bboxHelper(self,attr):
		if (len(attr) <= 1):
			attr = [attr.value[0][0]]
		else:
			attr = [self.file[attr.value[j].item()].value[0][0] for j in range(len(attr))]  
		return attr

# get_bbox returns a dict of data for the n(th) bbox. 
	def get_bbox(self,n):
		bbox = {}
		bb = self.digitStructBbox[n].item()
		bbox['height'] = self.bboxHelper(self.file[bb]["height"])
		bbox['label'] = self.bboxHelper(self.file[bb]["label"])
		bbox['left'] = self.bboxHelper(self.file[bb]["left"])
		bbox['top'] = self.bboxHelper(self.file[bb]["top"])
		bbox['width'] = self.bboxHelper(self.file[bb]["width"])
		return bbox

	def get_digit_structure(self,n):
		s = self.get_bbox(n)
		s['name']=self.get_name(n)
		return s

# get_all_digit_structure returns all the digitStruct from the input file.     
	def get_all_digit_structure(self):
		return [self.get_digit_structure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset.
	def get_all_digit_structure_by_digit(self):
		pictDat = self.get_all_digit_structure()
		result = []
		structCnt = 1
		for i in range(len(pictDat)):
			item = { 'filename' : pictDat[i]["name"] }
			figures = []
			for j in range(len(pictDat[i]['height'])):
			   figure = {}
			   figure['height'] = pictDat[i]['height'][j]
			   figure['label']  = pictDat[i]['label'][j]
			   figure['left']   = pictDat[i]['left'][j]
			   figure['top']    = pictDat[i]['top'][j]
			   figure['width']  = pictDat[i]['width'][j]
			   figures.append(figure)
			structCnt = structCnt + 1
			item['boxes'] = figures
			result.append(item)
		return np.array(result)
