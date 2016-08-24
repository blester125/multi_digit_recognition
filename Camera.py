from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from Download import download
from Extract import extract
from Visualize import display_example, display_processed_example, load_example
from Digit_Struct_File import DigitStructFile
from Generate_Dataset import generate_dataset
from Data_Split import split
from Save import save
from Preprocess import preprocess_camera, preprocess_file_image

import Analytics
import Network

CAMERA_MAX_HEIGHT = 480
CAMERA_MAX_WIDTH = 640

def download_data():
	url = 'http://ufldl.stanford.edu/housenumbers/'
	# Download datasets
	train_filename = download(url, 'train.tar.gz')
	test_filename = download(url, 'test.tar.gz')
	extra_filename = download(url, 'extra.tar.gz')
	return train_filename, test_filename, extra_filename

def extract_data(train_filename, test_filename, extra_filename):
	#extract datasets
	train_folder = extract(train_filename)
	test_folder = extract(test_filename)
	extra_folder = extract(extra_filename)
	return train_folder, test_folder, extra_folder

def process_and_visualize(train_folder="train", test_folder="test", extra_folder="extra", display=""):
	if not(
		os.path.exists("train") or 
		os.path.exists("test") or 
		os.path.exists("extra")
		):
		if not(
			os.path.exists("train.tar.gz") or 
			os.path.exists("test.tar.gz") or 
			os.path.exists("extra.tar.gz")
			):
			# No tar.gz files found, data must be downloaded
			tr, t, e = download_data()
		# no folders found, need to extract the tar.gz
		train_folder, test_folder, extra_folder = extract_data(tr, t, e)
	
	# Set sequence lengths to 0 so multiple runs do not add to old run totals
	Analytics.load()
	Analytics.sequence_lengths = {'train': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
								  'extra': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
								  'test' : {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}}
	Analytics.save()

	# Get the DigitStructs for Training Data
	fin = os.path.join(train_folder, 'digitStruct.mat')
	dsf = DigitStructFile(fin)
	print("Parsing the training data from the digitStruct.mat file")
	train_data = dsf.get_all_digit_structure_by_digit()
	print("Parsed training data")

	if display != "":
		print("Displaying Examples with bounding boxes")
		example_indeces = np.random.randint(0, len(train_data), size=5)
		examples = train_data[example_indeces]
		Analytics.load()
		Analytics.train_samples = examples
		Analytics.save()
		for e in examples:
			display_example(e, train_folder)
	
	# Preprocess Training data and fetch labels
	print("Generating data set and processing data.")
	train_dataset, train_labels = generate_dataset(train_data, train_folder)

	if display != "":
		print("Displaying examples of preprocessed images")
		examples = train_dataset[example_indeces]
		labels = train_labels[example_indeces]
		for e, l in zip(examples, labels):
			print("The Label for this is:", l)
			display_processed_example(e)

	#Delete things to free up space
	if display != "":
		del example_indeces
		del examples
		del labels
	del train_data

	#Repeat for Extra Data
	fin = os.path.join(extra_folder, 'digitStruct.mat')
	dsf = DigitStructFile(fin)
	print("Parsing the extra data from the digitStruct.mat file")
	extra_data = dsf.get_all_digit_structure_by_digit()
	print("Parsed extra data")

	# Preprocess extra data and fetch labels
	print("Generating data set and processing data.")
	extra_dataset, extra_labels = generate_dataset(extra_data, extra_folder)

	# Delete to free space
	del extra_data

	# Create the Training and Validation sets
	print("Creating the Training and Validation sets")
	train_dataset, train_labels, valid_dataset, valid_labels = split(train_dataset, 
																	 train_labels,
																	 extra_dataset,
																	 extra_labels)
	print("Finished creating the sets")

	# Delete to free space
	del extra_dataset
	del extra_labels

	# Create the Test data set
	fin = os.path.join(test_folder, 'digitStruct.mat')
	dsf = DigitStructFile(fin)
	print("Parsing the test data from the digitStruct.mat file")
	test_data = dsf.get_all_digit_structure_by_digit()
	print("Parsed test data")

	if display != "":	
		example_indeces = np.random.randint(0, len(test_data), size=5)
		examples = test_data[example_indeces]
		Analytics.load()
		Analytics.test_samples = examples
		Analytics.save()

	# Preprocess test data and fetch labels
	print("Generating data set and processing data.")
	test_dataset, test_labels = generate_dataset(test_data, test_folder)

	# Delete to free space
	del test_data
	del fin
	del dsf

	#Save Data to files
	save(
			train_dataset,
			train_labels,
			valid_dataset,
			valid_labels,
			test_dataset,
			test_labels
		)

	del train_dataset
	del train_labels
	del valid_dataset
	del valid_labels
	del test_dataset
	del test_labels
	
	Analytics.display()
	Analytics.save()

##
# Display an image with the label and the prodictions
def example_plot(net):
	plt.rcParams['figure.figsize'] = (20.0, 20.0)
	f, ax = plt.subplots(nrows=1, ncols=5)
	model_input, model_labels = generate_dataset(Analytics.test_samples, 'test')
	predictions = net.predict(model_input, False)
	for i, r in enumerate(Analytics.test_samples):
		im = load_example(r, 'test')
		house_num = ''
		for k in np.arange(model_labels[i,0]):
			house_num += str(model_labels[i,k+1])
		pred = ''
		for k in np.arange(len(predictions[1])):
			if predictions[i][k] != 10:
				pred += str(predictions[i][k])
		ax[i].axis('off')
		ax[i].set_title("Label: " + house_num + "\nPredict: " + pred, loc='center')
		ax[i].imshow(im)
	plt.show()

##
# Use the default camera to capture an image and feed it into the network
def use_camera(net):
	print("Press 'p' to Predict or 'q' to Quit.")
	# Size of the helper box to center numbers
	width = 300
	height = 200
	x, y, w, h = get_box(width, height)
	print(x, y, w, h)
	# Use opencv to get images from the first camera
	cap = cv2.VideoCapture(0)
	# Set the capture size
	cap.set(3, CAMERA_MAX_WIDTH);
	cap.set(4, CAMERA_MAX_HEIGHT);
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		# Draw a rectangle on the image
		# (0, 255, 0) is Green
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# Display the resulting frame
		cv2.imshow('frame',frame)
		# Press 's' to take an image and predict with it
		if cv2.waitKey(1) & 0xFF == ord('p'):
			image = preprocess_camera(frame, y, x, width, height)
			image = Network.reshape_input(image)
			net.predict(image)
		# press 'q' to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

##
# Load an image from a file and pass it to the network
def predict_image(net):
	print("Currently only .png files are supported.")
	user_input = raw_input("Enter the filename: ")
	try:
		img = mpimg.imread(user_input)
	except IOError:
		print("Unable to find file named", user_input)
		user_input = raw_input("Enter the filename: ")
		try:
			img = mpimg.imread(user_input)
		except IOError:
			print("Unable to find file named", user_input)
			print("Good-bye.")
			exit()
	image = preprocess_file_image(img)
	image = Network.reshape_input(image)
	net.predict(image)

##
# Get the x and y values to draw a box on the camera 
def get_box(width, height):
  y = (CAMERA_MAX_HEIGHT - height)/2
  x = (CAMERA_MAX_WIDTH - width)/2
  return int(x), int(y), width, height

##
# Get the user input and return it pased as a number
def get_user_input(prompt, error):
	flag = False
	while (flag == False):
		user_input = raw_input(prompt)
		try:
			user_input = int(user_input)
			flag = True
		except ValueError:
			print(error)
	return user_input

def main():
	Analytics.load()
	#Analytics.save()
	if not(
		os.path.exists("train_dataset.npy") or
		os.path.exists("train_labels.npy") or
		os.path.exists("valid_dataset.npy") or
		os.path.exists("valid_labels.npy") or
		os.path.exists("test_dataset.npy") or
		os.path.exists("test_labels.npy")
		):
		print("Data does not exist, downloading now.")
		tr, t, e = download_data()
		train_folder, test_folder, extra_folder = extract_data(tr, t, e)

	# create my network object and the Tensorflow graph for it
	net = Network.Network()
	quit = False
	# Loop options because Tensorflow takes so long to import
	while quit == False:
		print("1. Process The Datasets")
		print("2. Train the model")
		print("3. Display Analytics")
		print("4. Example Use")
		print("5. Use the model")
		print("6. Quit")
		user_input = get_user_input("Select a number: ", "Please enter a number")
		if user_input == 1:
			print("Press enter to just process the data")
			user_input = raw_input("Press y to visualize it also: ") 
			process_and_visualize(display=user_input)
		elif user_input == 2:
			user_input = get_user_input(
							"How many training steps? ", 
							"Please enter a number")
			net.set_num_steps(int(user_input))
			# Load in the data from the .npy files
			net.load_data()
			net.do_training()
		elif user_input == 3:
			Analytics.display()
		elif user_input == 4:
			try:
				net.load()
			except ValueError:
				print("Failed to load model from ", net.savepath)
				print("Did you train your model yet?")
				exit()
			example_plot(net)
		elif user_input == 5:
			# Restore graph from the savepath file
			try:
				net.load()
			except ValueError:
				print("Failed to load model from ", net.savepath)
				print("Did you train your model yet?")
				exit()
			print("1. Use a Camera.")
			print("2. Use an image file.")
			user_input = get_user_input("Select a number: ", "Please enter a number")
			if int(user_input) == 1:
				use_camera(net)
			else:
				predict_image(net)
		elif user_input == 6:
			quit = True

if __name__ == '__main__':
		main()
