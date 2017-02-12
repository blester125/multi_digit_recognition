from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

##
# Convolutions

def convolution2D(
		x, 
		output_depth,
		kernel_size,
		strides=[1, 1], 
		padding='SAME',
		name="convlution",
		phase_train=tf.constant(False),
		use_batch_norm=False,
		weight_decay=0.0,
		stddev=1e-1
		):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[-1]
		regularizer = lambda t: l2_loss(t, weight=weight_decay)
		kernel = tf.get_variable(
			"weights",
			[kernel_size[0],
			 kernel_size[1],
			 input_depth,
			 output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			regularizer=regularizer,
			dtype=x.dtype)
		conv = tf.nn.conv2d(x, kernel, [1, strides[0], strides[1], 1], padding=padding)
		if use_batch_norm:
			conv = batch_norm(conv, phase_train)
	return conv

##
# Regularization

def l2_loss(tensor, weight=1.0, name=None):
	with tf.name_scope(name):
		weight = tf.convert_to_tensor(weight, dtype=tensor.dtype.base_dtype, name='loss_weight')
		loss = tf.mul(weight, tf.nn.l2_loss(tensor), name='value')
	return loss

def batch_norm(x, phase_train, decay=0.5, epsilon=1e-3):
	with tf.variable_scope("Batch_Norm"):
		#phase_train = tf.convert_to_tensor(phase_train, dtype=tf.bool)
		n_out = int(x.get_shape()[-1])
		beta = tf.Variable(
					tf.constant(0.1, shape=[n_out], dtype=x.dtype), 
					name="beta", 
					trainable=True, 
					dtype=x.dtype
				)
		gamma = tf.Variable(
					tf.constant(0.1, shape=[n_out], dtype=x.dtype), 
					name="gamma", 
					trainable=True, 
					dtype=x.dtype
				)
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=decay)
		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
		mean, var = control_flow_ops.cond(phase_train,
										  mean_var_with_update,
										  lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
	return normed

def fully_connected(x, output_depth, name="fully_connected", weight_decay=0.0, stddev=1e-1):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[-1]
		regularizer = lambda t: l2_loss(t, weight=weight_decay)
		weights = tf.get_variable(
			"weights",
			[input_depth, output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			regularizer=regularizer,
			dtype=x.dtype)
		biases = tf.get_variable(
			initializer=tf.constant(0.1, shape=[output_depth]), 
			name="biases", 
			dtype=x.dtype)
		activation = tf.matmul(x, weights) + biases
		out = tf.nn.relu(activation)
	return out

##
# Outputs

def softmax(x, output_depth, name="softmax", stddev=1e-1):
	with tf.variable_scope(name):
		input_depth = x.get_shape()[-1]
		weights = tf.get_variable(
			"weights",
			[input_depth, output_depth],
			initializer=tf.truncated_normal_initializer(stddev=stddev),
			dtype=x.dtype)
		biases = tf.get_variable(
			initializer=tf.constant(0.1, shape=[output_depth]), 
			name="biases", 
			dtype=x.dtype)
		activation = tf.matmul(x, weights) + biases
		out = tf.nn.softmax(activation)
	return out

def load_minibatch(X_train, y_train, offset, batch_size):
	y_batch = np.array(y_train[offset:min(offset+batch_size, X_train.shape[0])])
	X_batch = np.array(X_train[offset:min(offset+batch_size, X_train.shape[0])])
	return X_batch, y_batch

def reshape_input(image):
	return np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])

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