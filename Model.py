from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import range

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Save import load
from Graph import Graph

from Network import load_minibatch

import Network
import Analytics

BATCH_SIZE = 64
NUM_LABELS = 11
NUM_LENGTHS = 7
NUM_CHANNELS = 1
NUM_LOGITS = 6
IMAGE_SIZE = 64
DEPTH1 = 16
DEPTH2 = 32
DEPTH3 = 64
DEPTH4 = 128
DEPTH5 = 256
NUM_HIDDEN1 = 256
NUM_HIDDEN2 = 128

# beta_regul = 1e-3

class Model():
	def __init__(self, batch_size=BATCH_SIZE, epochs=20, savepath="model.ckpt", logdir="log"):
		self.savepath = savepath
		self.logdir = logdir
		self.batch_size = batch_size
		self.epochs = epochs

		self.sess = tf.Session()
		self.x = tf.placeholder(
					tf.float32, 
					shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], 
					name="x")
		self.y = tf.placeholder(
					tf.int32, 
					shape=[None, NUM_LOGITS], 
					name="y")
		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		self.phase_train = tf.placeholder(tf.bool, name="phase_train")

		self.logits = inference(self.x, self.keep_prob, self.phase_train)
		self.loss = loss(self.logits, self.y)
		self.train_op = train(self.loss)
		self.prediction = prediction(self.logits)
		self.display_prediction = display_prediction(self.prediction)

		self.summary_op = tf.merge_all_summaries()

		self.saver = tf.train.Saver()
		self.writer = tf.train.SummaryWriter(logdir, graph=self.sess.graph)
		self.sess.run(tf.initialize_all_variables())

		self.graph = Graph(title="Accs", x_label="epoch", y_label="Acc")

	def load(self):
		self.saver.restore(self.sess, self.savepath)

	def load_data(self, dataset=1):
		(self.train_dataset, 
		 self.train_labels, 
		 self.valid_dataset, 
		 self.valid_labels, 
		 self.test_dataset, 
		 self.test_labels) = load(dataset)

	def do_training(self):
		self.train_accs = []
		self.valid_accs = []
		print("Training")
		with self.sess.as_default():
			start = time.time()
			for epoch in range(self.epochs):
				offset = 0
				#epoch_loss = 0
				while offset < self.train_dataset.shape[0]:
					X_batch, y_batch = load_minibatch(
											self.train_dataset,
											self.train_labels,
											offset,
											self.batch_size,
										)
					_, l, summary = self.sess.run(
										[self.train_op,
										 self.loss,
										 self.summary_op],
										 feed_dict = {
										 	self.x: X_batch,
										 	self.y: y_batch,
										 	self.keep_prob: 0.5,
										 	self.phase_train: True
										 })
					#epoch_loss += l * X_batch.get_shape().as_list()[0]
					offset = min(offset + self.batch_size, self.train_dataset.shape[0])
				train_pred = self.prediction.eval(
									feed_dict={
										self.x: X_batch,
										self.y: y_batch,
										self.keep_prob: 1.0,
										self.phase_train: False
									})

				train_acc = accuracy(train_pred, y_batch[:,1:6])
				print("Training Accuracy", train_acc, "at Epoch:", epoch)
				self.train_accs.append([epoch, train_acc])

			
				offset = 0
				valid_acc = 0
				count = 0
				while offset < self.valid_dataset.shape[0]:
					X_batch, y_batch = load_minibatch(
											self.valid_dataset,
											self.valid_labels,
											offset,
											self.batch_size
										)
					valid_pred = self.prediction.eval(
										feed_dict={
											self.x: X_batch, 
											self.y: y_batch, 
											self.keep_prob: 1.0,
											self.phase_train: False
										}
								)
					temp_valid_acc = accuracy(valid_pred, y_batch[:,1:6])
					old_offset = offset
					offset = min(offset + self.batch_size, self.valid_dataset.shape[0])
					count += 1
					#valid_acc = ((valid_acc * old_offset) + 
					#			 (temp_valid_acc * self.batch_size)/offset)
					valid_acc += temp_valid_acc
				valid_acc = valid_acc / count
				self.valid_accs.append([epoch, valid_acc])
				print("Valididation Accuracy", valid_acc, "at Epoch:", epoch)

				save_path = self.saver.save(self.sess, self.savepath)
				self.writer.add_summary(summary, epoch)
				
				# Graph of accuracies		
				self.graph.update([x[0] for x in self.train_accs], [y[1] for y in self.train_accs],"b", "Training")
				self.graph.update([x[0] for x in self.valid_accs], [y[1] for y in self.valid_accs], "r", "Validation")
				#if epoch == 0:
				#	self.graph.addLegend()
			print("Evaluating Test Dataset")
			offset = 0
			test_acc = 0
			count = 0
			while offset < self.test_dataset.shape[0]:
				X_batch, y_batch = load_minibatch(
										self.test_dataset,
										self.test_labels,
										offset,
										self.batch_size
									)
				test_pred = self.prediction.eval(
								feed_dict={
									self.x: X_batch, 
									self.y: y_batch,
									self.keep_prob: 1.0,
									self.phase_train: False
								}
							)
				temp_test_acc = accuracy(test_pred, y_batch[:, 1:6])
				old_offset = offset
				offset = min(offset + self.batch_size, self.test_dataset.shape[0])
				count += 1
				#test_acc = ((test_acc * old_offset) +
				#			(temp_test_acc * self.batch_size)/offset)
				test_acc += temp_test_acc
			test_acc = test_acc / count
			print("Test accuracy: %.1f%%" % (test_acc))

			Analytics.train_time = time.time() - start
			Analytics.train_accuracy = train_acc
			Analytics.valid_accuracy = valid_acc
			Analytics.test_accuracy = test_acc
			Analytics.display()
			Analytics.save()
			plt.show()

	def predict(self, image, print_=False):
		pred = self.sess.run(
					self.display_prediction, 
					feed_dict={self.x: image, self.keep_prob: 1.0}
			   )
		if print_ == True:
			self.print_prediction(pred)
		return pred

	def print_prediction(self, pred):
		for i in range(pred.shape[0]):
			value = ""
			for j in range(pred[i].shape[0]):
				if pred[i][j] != 10:
					value += str(pred[i][j])
			print(value)

def accuracy(predictions, labels):
	return (
			100.0 * 
			np.sum(np.argmax(predictions, 2).T == labels) / 
					predictions.shape[1] / 
					predictions.shape[0]
			)

def accuracy2(predictions, labels): 
	value = tf.reduce_sum(tf.cast(tf.equal(tf.transpose(tf.argmax(predictions, 2)), tf.cast(labels[:, 1:6], tf.int64)), tf.int64)) 
	value = tf.cast(value, tf.float32)
	value = tf.div(value, tf.cast([predictions.get_shape().as_list()[0]],tf.float32))
	value = tf.div(value, tf.cast([predictions.get_shape().as_list()[1]],tf.float32))
	value = tf.mul(value, [100.0])
	return value

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

def inference(images, keep_prob, phase_train):
	conv1 = Network.convolution2D(
				images, 
				DEPTH1, 
				[5, 5], 
				padding="VALID", 
				phase_train=phase_train, 
				use_batch_norm=True,
				weight_decay=0.04, 
				name="Conv1")
	h_conv1 = tf.nn.relu(conv1)
	print(h_conv1.get_shape().as_list())

	pool1 = Network.max_pool_2x2(h_conv1, 1)
	print(pool1.get_shape().as_list())

	dropout1 = tf.nn.dropout(pool1, keep_prob, name="dropout1")

	conv2 = Network.convolution2D(
				dropout1,
				DEPTH2,
				[1, 1],
				padding="VALID",
				phase_train=phase_train,
				use_batch_norm=True,
				weight_decay=0.04,
				name="Conv2")
	h_conv2 = tf.nn.relu(conv2)
	print(h_conv2.get_shape().as_list())
	conv3 = Network.convolution2D(
				h_conv2,
				DEPTH3,
				[3, 3],
				padding="VALID",
				phase_train=phase_train,
				use_batch_norm=True,
				weight_decay=0.04,
				name="Conv3")
	h_conv3 = tf.nn.relu(conv3)
	print(h_conv3.get_shape().as_list())

	pool2 = Network.max_pool_2x2(h_conv3, 2)
	print(pool2.get_shape().as_list())

	dropout2 = tf.nn.dropout(pool2, keep_prob, name="dropout2")

	conv4 = Network.convolution2D(
				dropout2,
				DEPTH4,
				[3, 3],
				padding="VALID",
				phase_train=phase_train,
				use_batch_norm=True,
				weight_decay=0.04,
				name="Conv4")
	h_conv4 = tf.nn.relu(conv4)
	print(h_conv4.get_shape().as_list())

	pool3 = Network.max_pool_2x2(h_conv4, 3)
	print(pool3.get_shape().as_list())

	dropout3 = tf.nn.dropout(pool3, keep_prob, name="dropout3")

	conv5 = Network.convolution2D(
				dropout3,
				DEPTH5,
				[3, 3],
				padding="VALID",
				phase_train=phase_train,
				use_batch_norm=True,
				weight_decay=0.04,
				name="Conv5")
	h_conv5 = tf.nn.relu(conv5)
	print(h_conv5.get_shape().as_list())

	pool4 = Network.max_pool_2x2(h_conv5, 4)
	print(pool4.get_shape().as_list())

	pool5 = Network.max_pool_2x2(pool4, 5)
	print(pool5.get_shape().as_list())

	flatten = tf.reshape(pool5, [-1, DEPTH5])
	print(flatten.get_shape().as_list())

	hidden1 = Network.fully_connected(
				flatten,
				NUM_HIDDEN1,
				weight_decay=0.04,
				name="hidden1")
	hidden2 = Network.fully_connected(
				flatten,
				NUM_HIDDEN1,
				weight_decay=0.04,
				name="hidden2")
	hidden3 = Network.fully_connected(
				flatten,
				NUM_HIDDEN1,
				weight_decay=0.04,
				name="hidden3")
	hidden4 = Network.fully_connected(
				flatten,
				NUM_HIDDEN1,
				weight_decay=0.04,
				name="hidden4")
	hidden5 = Network.fully_connected(
				flatten,
				NUM_HIDDEN1,
				weight_decay=0.04,
				name="hidden5")
	hidden6 = Network.fully_connected(
				flatten,
				NUM_HIDDEN1,
				weight_decay=0.04,
				name="hidden6")

	dropout_h_1 = tf.nn.dropout(hidden1, keep_prob, name="dropout_h_1")
	dropout_h_2 = tf.nn.dropout(hidden2, keep_prob, name="dropout_h_2")
	dropout_h_3 = tf.nn.dropout(hidden3, keep_prob, name="dropout_h_3")
	dropout_h_4 = tf.nn.dropout(hidden4, keep_prob, name="dropout_h_4")
	dropout_h_5 = tf.nn.dropout(hidden5, keep_prob, name="dropout_h_5")
	dropout_h_6 = tf.nn.dropout(hidden6, keep_prob, name="dropout_h_6")
	
	logit1 = Network.fully_connected(
				dropout_h_1,
				NUM_LENGTHS,
				weight_decay=0.04,
				name="logit1")
	logit2 = Network.fully_connected(
				dropout_h_2,
				NUM_LABELS,
				weight_decay=0.04,
				name="logit2")
	logit3 = Network.fully_connected(
				dropout_h_3,
				NUM_LABELS,
				weight_decay=0.04,
				name="logit3")
	logit4 = Network.fully_connected(
				dropout_h_4,
				NUM_LABELS,
				weight_decay=0.04,
				name="logit4")
	logit5 = Network.fully_connected(
				dropout_h_5,
				NUM_LABELS,
				weight_decay=0.04,
				name="logit5")
	logit6 = Network.fully_connected(
				dropout_h_6,
				NUM_LABELS,
				weight_decay=0.04,
				name="logit6")
	return [logit1, logit2, logit3, logit4, logit5, logit6]
	
def loss(logits, labels):
	loss_per_digit = [tf.reduce_mean(
							tf.nn.sparse_softmax_cross_entropy_with_logits(
							logits[i],
							labels[:,i]
						))
						for i in range(NUM_LOGITS)]
	loss_value = tf.add_n(loss_per_digit)
	tf.scalar_summary('loss', loss_value)
	return loss_value

def train(total_loss):
	with tf.variable_scope('train') as scope:
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
		optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
	return optimizer

def prediction(logits):
	with tf.variable_scope('prediction') as scope:
		pred = tf.pack([tf.nn.softmax(logits[1]),\
						tf.nn.softmax(logits[2]),\
						tf.nn.softmax(logits[3]),\
						tf.nn.softmax(logits[4]),\
						tf.nn.softmax(logits[5])])
	return pred

def display_prediction(prediction):
	with tf.variable_scope('display') as scope:
		return tf.transpose(tf.argmax(prediction, 2))

if __name__ == "__main__":
	model = Model(epochs=60)
	model.load_data(5)
	model.do_training()

# def inference(images, keep_prob):
# 	tf.image_summary('input', images, 5)

# 	with tf.variable_scope('model') as scope:
# 		##
# 		# images are size: [? x 64 x 64 x 1]
		
# 		##
# 		# Convolutions
# 		# First Layer
# 		with tf.variable_scope('conv1') as scope:
# 			weights = tf.get_variable(
# 						shape=[5, 5, NUM_CHANNELS, DEPTH1],
# 						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[DEPTH1]), 
# 						name='biases')
# 			conv = conv2d(images, weights)
# 			bias = tf.nn.bias_add(conv, biases)
# 			conv1 = tf.nn.relu(bias, name=scope.name)
# 			tf.histogram_summary("conv_1_weights", weights)
# 			tf.image_summary('convultion1', conv1[:, :, :, 0:1], 5)
# 		##
# 		# images are size: [? x 60 x 60 x 16]
# 		print(conv1.get_shape().as_list())
# 		# Pooling
# 		pool1 = max_pool_2x2(conv1, 1)
# 		##
# 		# images are size: [? x 30 x 30 x 16]
# 		print(pool1.get_shape().as_list())
# 		# Normialization
# 		norm1 = normalize(pool1, 1)
# 		# Dropout
# 		dropout1 = tf.nn.dropout(norm1, keep_prob, name='dropout1')


# 		# Second Layer
# 		with tf.variable_scope('conv2') as scope:
# 			weights = tf.get_variable(
# 						shape=[1, 1, DEPTH1, DEPTH2],
# 						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[DEPTH2]),
# 						name='biases')
# 			conv = conv2d(dropout1, weights)
# 			bias = tf.nn.bias_add(conv, biases)
# 			conv2 = tf.nn.relu(bias, name=scope.name)
# 			tf.histogram_summary("conv_2_weights", weights)
# 			tf.image_summary('convultion2', conv2[:, :, :, 0:1], 5)
# 		##
# 		# images are size: [? x 30 x 30 x 32]
# 		print(conv2.get_shape().as_list())
# 		with tf.variable_scope('conv2_2') as scope:
# 			weights = tf.get_variable(
# 						shape=[3, 3, DEPTH2, DEPTH3],
# 						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[DEPTH3]),
# 						name='biases')
# 			conv = conv2d(conv2, weights)
# 			bias = tf.nn.bias_add(conv, biases)
# 			conv2 = tf.nn.relu(bias, name=scope.name)
# 		##
# 		# images are size: [? x 28 x 28 x 64]
# 		print(conv2.get_shape().as_list())
# 		# Normalization
# 		norm2 = normalize(conv2, 2)
# 		# Pooling
# 		pool2 = max_pool_2x2(norm2, 2)
# 		##
# 		# images are size: [? x 14 x 14 x 64]
# 		print(pool2.get_shape().as_list())
# 		# Dropout
# 		dropout2 = tf.nn.dropout(pool2, keep_prob, name='dropout2')


# 		# Thrid Layer
# 		with tf.variable_scope('conv3') as scope:
# 			weights = tf.get_variable(
# 						shape=[3, 3, DEPTH3, DEPTH4],
# 						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[DEPTH4]),
# 						name='biases')
# 			conv = conv2d(dropout2, weights)
# 			bias = tf.nn.bias_add(conv, biases)
# 			conv3 = tf.nn.relu(bias, name=scope.name)
# 			tf.histogram_summary("conv_3_weights", weights)
# 			tf.image_summary('convultion3', conv3[:, :, :, 0:1], 5)
# 		##
# 		# images are size: [? x 12 x 12 x 128]
# 		print(conv3.get_shape().as_list())
# 		# Pooling
# 		pool3 = max_pool_2x2(conv3, 3)
# 		##
# 		# images are size: [? x 6 x 6 x 128]
# 		print(pool3.get_shape().as_list())
# 		# Normalization
# 		norm3 = normalize(pool3, 3)
# 		# Dropout
# 		dropout3 = tf.nn.dropout(norm3, keep_prob, name='dropout3')


# 		# Fourth Layer
# 		with tf.variable_scope('conv4') as scope:
# 			weights = tf.get_variable(
# 						shape=[3, 3, DEPTH4, DEPTH5],
# 						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[DEPTH5]),
# 						name='biases')
# 			conv = conv2d(dropout3, weights)
# 			bias = tf.nn.bias_add(conv, biases)
# 			tf.histogram_summary("conv_4_weights", weights)
# 			conv4 = tf.nn.relu(bias, name=scope.name)
# 		##
# 		# images are size: [? x 4 x 4 x 256]
# 		print(conv4.get_shape().as_list())
# 		# Normalization
# 		norm4 = normalize(conv4, 4)
# 		# Dropout
# 		norm4 = tf.nn.dropout(norm4, keep_prob, name='dropout4')

# 		pool4 = max_pool_2x2(conv4, 4)
# 		##
# 		# images are size [? x 2 x 2 x 256]
# 		print(pool4.get_shape().as_list())
# 		pool5 = max_pool_2x2(pool4, 5)
# 		##
# 		# images are size [? x 1 x 1 x 256]
# 		print(pool5.get_shape().as_list())
# 		# Move everything into depth
# 		reshape = tf.reshape(pool5, [-1, DEPTH5])
# 		##
# 		# images are size: [? x 256]

# 		##
# 		# Hidden Layers
# 		with tf.variable_scope('hidden1') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
# 						name='biases')
# 			activation = tf.matmul(reshape, weights) + biases
# 			hidden1 = tf.nn.relu(activation, name=scope.name)
# 			tf.histogram_summary("hidden_1_weights", weights)
# 			hidden1 = tf.nn.dropout(hidden1, keep_prob, name='dropout')

# 		with tf.variable_scope('hidden2') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
# 						name='biases')
# 			activation = tf.matmul(reshape, weights) + biases
# 			hidden2 = tf.nn.relu(activation, name=scope.name)
# 			tf.histogram_summary("hidden_2_weights", weights)
# 			hidden2 = tf.nn.dropout(hidden2, keep_prob, name='dropout')

# 		with tf.variable_scope('hidden3') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
# 						name='biases')
# 			activation = tf.matmul(reshape, weights) + biases
# 			hidden3 = tf.nn.relu(activation, name=scope.name)
# 			tf.histogram_summary("hidden_3_weights", weights)
# 			hidden3 = tf.nn.dropout(hidden3, keep_prob, name='dropout')

# 		with tf.variable_scope('hidden4') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
# 						name='biases')
# 			activation = tf.matmul(reshape, weights) + biases
# 			hidden4 = tf.nn.relu(activation, name=scope.name)
# 			tf.histogram_summary("hidden_4_weights", weights)
# 			hidden4 = tf.nn.dropout(hidden4, keep_prob, name='dropout')

# 		with tf.variable_scope('hidden5') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
# 						name='biases')
# 			activation = tf.matmul(reshape, weights) + biases
# 			hidden5 = tf.nn.relu(activation, name=scope.name)
# 			tf.histogram_summary("hidden_5_weights", weights)
# 			hidden5 = tf.nn.dropout(hidden5, keep_prob, name='dropout')

# 		with tf.variable_scope('hidden6') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
# 						name='biases')
# 			activation = tf.matmul(reshape, weights) + biases
# 			hidden6 = tf.nn.relu(activation, name=scope.name)
# 			tf.histogram_summary("hidden_6_weights", weights)
# 			hidden6 = tf.nn.dropout(hidden6, keep_prob, name='dropout')

# 		## insert second hidden layer? NUM_HIDDEN1 = 256
# 		#                              NUM_HIDDEN2 = 128
# 		#                              NUM_HIDDEN3 = 16
# 		#                              NUM_LENGTHS = 6
# 		#                              NUM_LABELS = 11

# 		##
# 		# Logits layers
# 		with tf.variable_scope('logits1') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN2, NUM_LENGTHS],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_LENGTHS]),
# 						name='biases')
# 			tf.histogram_summary("logits_1_weights", weights)
# 			logits1 = tf.matmul(hidden1, weights) + biases

# 		with tf.variable_scope('logits2') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN2, NUM_LABELS],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
# 						name='biases')
# 			tf.histogram_summary("logits_2_weights", weights)
# 			logits2 = tf.matmul(hidden2, weights) + biases

# 		with tf.variable_scope('logits3') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN2, NUM_LABELS],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
# 						name='biases')
# 			tf.histogram_summary("logits_3_weights", weights)
# 			logits3 = tf.matmul(hidden3, weights) + biases

# 		with tf.variable_scope('logits4') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN2, NUM_LABELS],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
# 						name='biases')
# 			tf.histogram_summary("logits_4_weights", weights)
# 			logits4 = tf.matmul(hidden4, weights) + biases

# 		with tf.variable_scope('logits5') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN2, NUM_LABELS],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
# 						name='biases')
# 			tf.histogram_summary("logits_5_weights", weights)
# 			logits5 = tf.matmul(hidden5, weights) + biases

# 		with tf.variable_scope('logits6') as scope:
# 			weights = tf.get_variable(
# 						shape=[NUM_HIDDEN2, NUM_LABELS],
# 						initializer=tf.contrib.layers.xavier_initializer(),
# 						name='weights')
# 			biases = tf.get_variable(
# 						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
# 						name='biases')
# 			tf.histogram_summary("logits_6_weights", weights)
# 			logits6 = tf.matmul(hidden6, weights) + biases

# 		return [logits1, logits2, logits3, logits4, logits5, logits6]

