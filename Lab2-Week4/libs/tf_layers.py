import tensorflow as tf

def FullyConnected(input_tensor, num_inputs, num_outputs, name=None, initializer=tf.glorot_uniform_initializer()):
	"""
	Handy wrapper function for convolutional networks.

	Performs an affine layer (fully-connected) on the input tensor.
	"""
	shape = [num_inputs, num_outputs]

	# initialize weights and biases of the affine layer
	W = tf.get_variable(name+'.W' ,shape=shape, initializer=initializer)
	b = tf.get_variable(name+'.b', shape=shape[-1], initializer=initializer)

	fc = tf.matmul(input_tensor, W, name=name)
	fc = tf.nn.bias_add(fc, b, name=name+'_bias')

	return fc


def BatchNormalization(input_tensor, phase, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs batch normalization on the input tensor.
	"""
	normed = tf.contrib.layers.batch_norm(input_tensor, center=True, scale=True, is_training=phase, scope=name)
	return normed

def Conv2D(input_tensor, input_shape, filter_size, num_filters, strides=1, name=None, initializer=tf.glorot_uniform_initializer()):
	"""
	Handy helper function for convnets.

	Performs 2D convolution with a default stride of 1. The kernel has shape
	filter_size x filter_size with num_filters output filters.
	"""
	shape = [filter_size, filter_size, input_shape, num_filters]

	# initialize weights and biases of the convolution
	W = tf.get_variable(name+'.W' ,shape=shape, initializer=initializer)
	b = tf.get_variable(name+'.b', shape=shape[-1], initializer=initializer)

	conv = tf.nn.conv2d(input_tensor, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
	conv = tf.nn.bias_add(conv, b, name=name+'_bias')
	return conv

def MaxPooling2D(input_tensor, k=2, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs 2D max pool with a default stride of 2.
	"""
	pool = tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

	return pool


def AveragePooling2D(input_tensor, k=2, name=None):
	"""
	Handy wrapper function for convolutional networks.

	Performs 2D max pool with a default stride of 2.
	"""
	pool = tf.nn.avg_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

	return pool