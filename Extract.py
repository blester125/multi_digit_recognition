from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tarfile

##
# Adapted from the code in Udacity Deep Leanring assignments
# https://www.udacity.com/course/deep-learning--ud730
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity

def extract(filename, force=False):
	root = os.path.splitext(filename)[0]
	root = os.path.splitext(root)[0]
	if os.path.isdir(root) and not force:
		print("%s already present - Skipping extraction of %s." 
				% (root, filename))
	else:
		print("Extracting data for %s. This may take a while. Please wait." 
				% root)
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	print(root)
	return root	

if __name__ == '__main__':
	extract('train.tar.gz', True)
