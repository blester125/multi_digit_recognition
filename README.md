# Deep Learning for Image Processing
##

![Example Output](latex/images/example_output.png)

This project detects multidigit sequences in natural scenes.
The dataset for this project can be found at 
http://ufldl.stanford.edu/housenumbers/

This project has been updated from when I created it as the Machine 
Learning Capstone project for Udacity. I have added Batch normalization 
and a deeper network as well as creating 5 data points from each picture.
The write-up included is out of date. 

The model gets 98 precent training accuracy, 94 precent validation accuracy, and 95 precent test accurcay after 100 epochs. The model included Batch Norm, Dropout, and Weight Decay for regularization. Using so much regularization and still having so much difference between the training and validation error means that the training distribuation is very different than the validation distribution. To solve this more data is most likely needed.

The following libraries are required:

 * Tensorflow 0.9.0
 * numpy 1.10.4
 * scipy 0.17.0
 * matplotlib 1.5.1
 * PIL 1.1.7
 * opencv 3.1.0

The program is run with `python Camera.py`
To download and process the data run the first option "Process The Datasets"
This will download, extract, and process the images. It will save the data 
into the files "train_dataset.npy", "train_labels.npy", "valid_dataset.npy",
"valid_labels.npy", test_dataset.npy", and "test_labels.py".
The next step it to train the model using the second option "Train the
 model". This will train the model and save it into a file called 
model.ckpt. Analytics from the datasets and the training can be found 
using the third option "Display Analytics". The fouth option "Example Use" 
displays some example images along with the labels and predictions.
The fifth option "Use the model" allows users to use the camera or to 
load an image from disk that will be feed to the network.

The trained model is included (along with the file 
that includes the Analytics and a test folder to allow for seeing some 
images in the example use) so users can run the Example use case or the 
camera without training the model. Before training the model the data 
will have to be fetched which should run automatically when running Camera.py
