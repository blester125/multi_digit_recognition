import numpy as np

def load_minibatch(X_train, y_train, offset, batch_size):
	y_batch = np.array(y_train[offset:min(offset+batch_size, X_train.shape[0])])
	X_batch = np.array(X_train[offset:min(offset+batch_size, X_train.shape[0])])
	return X_batch, y_batch

def reshape_input(image):
	return np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])