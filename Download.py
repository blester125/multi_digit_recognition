from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

from six.moves.urllib.request import urlretrieve

last_percent_reported = None

def download_hook(count, blockSize, totalSize):
	global last_percent_reported
	percent = int(count * blockSize * 100 / totalSize)
	if last_percent_reported != percent:
		if percent % 5 == 0:
			sys.stdout.write("%s%%" % percent)
			sys.stdout.flush()
		else:
			sys.stdout.write(".")
			sys.stdout.flush()
	last_percent_reported = percent

def download(url, filename, force=False):
	if force or not os.path.exists(filename):
		print("Attempting to download:", filename)
		filename, _ = urlretrieve(
						url + "/" + filename, 
						filename, 
						reporthook=download_hook)
		print("\nDownload Complete")
	else:
		print("File already present, not downloading.")
	return filename

if __name__ == "__main__":
	test = download("http://ufldl.stanford.edu/housenumbers/", "test.tar.gz", force=True)