from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import range

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Save import load

import Analytics

BATCH_SIZE = 64
NUM_LABELS = 11
NUM_LENGTHS = 7
NUM_CHANNELS = 1
NUM_LOGITS = 6
IMAGE_SIZE = 50
DEPTH1 = 16
DEPTH2 = 32
DEPTH3 = 64
DEPTH4 = 128
NUM_HIDDEN1 = 128
NUM_HIDDEN2 = 16
NUM_STEPS = 100001

# beta_regul = 1e-3

class Network():
	def __init__(self, num_steps=NUM_STEPS, savepath="model.ckpt", logdir="log"):
		self.savepath = savepath
		self.logdir = logdir
		self.num_steps = num_steps
		
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

		self.logits = inference(self.x, self.keep_prob)
		self.loss = loss(self.logits, self.y)
		self.train_op = train(self.loss)
		self.prediction = prediction(self.logits)
		#self.accuracy = accuracy2(self.prediction, self.y)
		self.display_prediction = display_prediction(self.prediction)

		self.summary_op = tf.merge_all_summaries()

		self.saver = tf.train.Saver()
		self.writer = tf.train.SummaryWriter(logdir, graph=self.sess.graph)

		self.sess.run(tf.initialize_all_variables())

	def load(self):
		self.saver.restore(self.sess, self.savepath)

	def load_data(self):
		self.train_dataset, \
		self.train_labels, \
		self.valid_dataset, \
		self.valid_labels, \
		self.test_dataset, \
		self.test_labels = load()

	def do_training(self):
		self.train_accs = []
		self.valid_accs = []
		print("Training")
		with self.sess.as_default():
			start = time.time()
			for step in range(self.num_steps):
				offset = ((step * BATCH_SIZE) % 
						  (self.train_labels.shape[0] - BATCH_SIZE))
				batch_data = self.train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
				batch_labels = self.train_labels[offset:(offset + BATCH_SIZE),:]
				feed_dict = {
						self.x: batch_data, 
						self.y: batch_labels, 
						self.keep_prob: .9375
					}
				_, l, summary = self.sess.run(
							[self.train_op, self.loss, self.summary_op], 
							feed_dict=feed_dict)

				if (step % 500 == 0): 
					print("Minibatch loss at step %d: %f" % (step, l))
					train_pred = self.prediction.eval(
										feed_dict={
											self.x: batch_data, 
											self.y: batch_labels, 
											self.keep_prob: 1.0
										}
								)
					train_acc = accuracy(train_pred, batch_labels[:,1:6])
					self.train_accs.append([step, train_acc])
					print("Minibatch accuracy: %.1f%%" % train_acc)
					valid_pred = self.prediction.eval(
										feed_dict={
											self.x: self.valid_dataset, 
											self.y: self.valid_labels, 
											self.keep_prob: 1.0
										}
								)
					valid_acc = accuracy(valid_pred, self.valid_labels[:,1:6])
					self.valid_accs.append([step, valid_acc])
					print("Validation accuracy: %.1f%%" % valid_acc)
					save_path = self.saver.save(self.sess, self.savepath)
					self.writer.add_summary(summary, step)
			# Graph of accuracies		
			plt.plot(
				[x[0] for x in self.train_accs], 
				[y[1] for y in self.train_accs],
				'b',
				[a[0] for a in self.valid_accs], 
				[b[1] for b in self.valid_accs],
				'r')
			plt.title("Training and Validation Accuracy")

			test_pred = self.prediction.eval(
								feed_dict={
									self.x: self.test_dataset, 
									self.keep_prob: 1.0
								}
							)
			test_acc = accuracy(test_pred, self.test_labels[:,1:6])
			print("Test accuracy: %.1f%%" % test_acc)

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

	def set_num_steps(self, num):
		self.num_steps = num

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

def reshape_input(image):
	return np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')

def max_pool_2x2(x, number):
	return tf.nn.max_pool(
				x, 
				ksize=[1,2,2,1], 
				strides=[1,2,2,1], 
				padding='SAME',
				name='pool' + str(number))

def normalize(x, number):
	return tf.nn.local_response_normalization(
				x,
				bias=1.0,
				alpha = 0.001 / 9.0,
				beta = 0.75,
				name='norm' + str(number))

def inference(images, keep_prob):
	tf.image_summary('input', images, 5)

	with tf.variable_scope('model') as scope:
		##
		# images are size: [? x 50 x 50 x 1]
		
		##
		# Convolutions
		# First Layer
		with tf.variable_scope('conv1') as scope:
			weights = tf.get_variable(
						shape=[3, 3, NUM_CHANNELS, DEPTH1],
						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH1]), 
						name='biases')
			conv = conv2d(images, weights)
			bias = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(bias, name=scope.name)
			tf.histogram_summary("conv_1_weights", weights)
			tf.image_summary('convultion1', conv1[:, :, :, 0:1], 5)
		##
		# images are size: [? x 48 x 48 x 16]

		# Pooling
		pool1 = max_pool_2x2(conv1, 1)
		##
		# images are size: [? x 24 x 24 x 16]

		# Normialization
		norm1 = normalize(pool1, 1)
		# Dropout
		dropout1 = tf.nn.dropout(norm1, keep_prob, name='dropout1')


		# Second Layer
		with tf.variable_scope('conv2') as scope:
			weights = tf.get_variable(
						shape=[1, 1, DEPTH1, DEPTH2],
						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH2]),
						name='biases')
			conv = conv2d(dropout1, weights)
			bias = tf.nn.bias_add(conv, biases)
			conv2 = tf.nn.relu(bias, name=scope.name)
			tf.histogram_summary("conv_2_weights", weights)
			tf.image_summary('convultion2', conv2[:, :, :, 0:1], 5)
		##
		# images are size: [? x 24 x 24 x 32]

		# Normalization
		norm2 = normalize(conv2, 2)
		# Pooling
		pool2 = max_pool_2x2(norm2, 2)
		##
		# images are size: [? x 12 x 12 x 32]

		# Dropout
		dropout2 = tf.nn.dropout(pool2, keep_prob, name='dropout2')


		# Thrid Layer
		with tf.variable_scope('conv3') as scope:
			weights = tf.get_variable(
						shape=[5, 5, DEPTH2, DEPTH3],
						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH3]),
						name='biases')
			conv = conv2d(dropout2, weights)
			bias = tf.nn.bias_add(conv, biases)
			conv3 = tf.nn.relu(bias, name=scope.name)
			tf.histogram_summary("conv_3_weights", weights)
			tf.image_summary('convultion3', conv3[:, :, :, 0:1], 5)
		##
		# images are size: [? x 8 x 8 x 64]

		# Pooling
		pool3 = max_pool_2x2(conv3, 3)
		##
		# images are size: [? x 4 x 4 x 64]

		# Normalization
		norm3 = normalize(pool3, 3)
		# Dropout
		dropout3 = tf.nn.dropout(norm3, keep_prob, name='dropout3')


		# Fourth Layer
		with tf.variable_scope('conv4') as scope:
			weights = tf.get_variable(
						shape=[4, 4, DEPTH3, DEPTH4],
						initializer=tf.contrib.layers.xavier_initializer_conv2d(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[DEPTH4]),
						name='biases')
			conv = conv2d(dropout3, weights)
			bias = tf.nn.bias_add(conv, biases)
			tf.histogram_summary("conv_4_weights", weights)
			conv4 = tf.nn.relu(bias, name=scope.name)
		##
		# images are size: [? x 1 x 1 x 128]

		# Normalization
		norm4 = normalize(conv4, 4)
		# Dropout
		norm4 = tf.nn.dropout(norm4, keep_prob, name='dropout4')

		# Move everything into depth
		reshape = tf.reshape(norm4, [-1, 128])
		##
		# images are size: [? x 128]

		##
		# Hidden Layers
		with tf.variable_scope('hidden1') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
						name='biases')
			activation = tf.matmul(reshape, weights) + biases
			hidden1 = tf.nn.relu(activation, name=scope.name)
			tf.histogram_summary("hidden_1_weights", weights)
			hidden1 = tf.nn.dropout(hidden1, keep_prob, name='dropout')

		with tf.variable_scope('hidden2') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
						name='biases')
			activation = tf.matmul(reshape, weights) + biases
			hidden2 = tf.nn.relu(activation, name=scope.name)
			tf.histogram_summary("hidden_2_weights", weights)
			hidden2 = tf.nn.dropout(hidden2, keep_prob, name='dropout')

		with tf.variable_scope('hidden3') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
						name='biases')
			activation = tf.matmul(reshape, weights) + biases
			hidden3 = tf.nn.relu(activation, name=scope.name)
			tf.histogram_summary("hidden_3_weights", weights)
			hidden3 = tf.nn.dropout(hidden3, keep_prob, name='dropout')

		with tf.variable_scope('hidden4') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
						name='biases')
			activation = tf.matmul(reshape, weights) + biases
			hidden4 = tf.nn.relu(activation, name=scope.name)
			tf.histogram_summary("hidden_4_weights", weights)
			hidden4 = tf.nn.dropout(hidden4, keep_prob, name='dropout')

		with tf.variable_scope('hidden5') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
						name='biases')
			activation = tf.matmul(reshape, weights) + biases
			hidden5 = tf.nn.relu(activation, name=scope.name)
			tf.histogram_summary("hidden_5_weights", weights)
			hidden5 = tf.nn.dropout(hidden5, keep_prob, name='dropout')

		with tf.variable_scope('hidden6') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN1, NUM_HIDDEN2],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_HIDDEN2]),
						name='biases')
			activation = tf.matmul(reshape, weights) + biases
			hidden6 = tf.nn.relu(activation, name=scope.name)
			tf.histogram_summary("hidden_6_weights", weights)
			hidden6 = tf.nn.dropout(hidden6, keep_prob, name='dropout')

		##
		# Logits layers
		with tf.variable_scope('logits1') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN2, NUM_LENGTHS],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_LENGTHS]),
						name='biases')
			tf.histogram_summary("logits_1_weights", weights)
			logits1 = tf.matmul(hidden1, weights) + biases

		with tf.variable_scope('logits2') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN2, NUM_LABELS],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
						name='biases')
			tf.histogram_summary("logits_2_weights", weights)
			logits2 = tf.matmul(hidden2, weights) + biases

		with tf.variable_scope('logits3') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN2, NUM_LABELS],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
						name='biases')
			tf.histogram_summary("logits_3_weights", weights)
			logits3 = tf.matmul(hidden3, weights) + biases

		with tf.variable_scope('logits4') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN2, NUM_LABELS],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
						name='biases')
			tf.histogram_summary("logits_4_weights", weights)
			logits4 = tf.matmul(hidden4, weights) + biases

		with tf.variable_scope('logits5') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN2, NUM_LABELS],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
						name='biases')
			tf.histogram_summary("logits_5_weights", weights)
			logits5 = tf.matmul(hidden5, weights) + biases

		with tf.variable_scope('logits6') as scope:
			weights = tf.get_variable(
						shape=[NUM_HIDDEN2, NUM_LABELS],
						initializer=tf.contrib.layers.xavier_initializer(),
						name='weights')
			biases = tf.get_variable(
						initializer=tf.constant(1.0, shape=[NUM_LABELS]),
						name='biases')
			tf.histogram_summary("logits_6_weights", weights)
			logits6 = tf.matmul(hidden6, weights) + biases

		return [logits1, logits2, logits3, logits4, logits5, logits6]

# def loss(logits, labels):
# 	#labels = tf.cast(labels, tf.int64)
# 	total_loss = []
# 	for i in range(len(logits)):
# 		total_loss.append( tf.reduce_mean(
# 					tf.nn.sparse_softmax_cross_entropy_with_logits(
# 						logits[i], 
# 						labels[:,i]
# 					)
# 				))
# 	return tf.add_n(total_loss)
	
def loss(logits, labels):
	with tf.variable_scope('loss') as scope:
		loss_value = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[0], labels[:,0])) +\
					 tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[1], labels[:,1])) +\
					 tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[2], labels[:,2])) +\
					 tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[3], labels[:,3])) +\
					 tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[4], labels[:,4])) +\
					 tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[5], labels[:,5]))
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