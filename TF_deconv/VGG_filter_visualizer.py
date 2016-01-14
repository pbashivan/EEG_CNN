"""Builds the EEG-CNN network, initializes it with the parameters from the trained network and uses deconvolutional
neural networks to project the feature maps into the image space.
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import numpy as np
import scipy.io
import seaborn as sb
import matplotlib.pyplot as pl
sb.set_style('white')
sb.set_context('talk')
sb.set(font_scale=0.5)
from get_projection import vis_square

import tensorflow.python.platform
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10_input
from tensorflow.python.platform import gfile

# FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 128,
#                             """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                            """Path to the CIFAR-10 data directory.""")

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

filename = '../EEG_images_32_timeWin'   # filename for EEG-images
subjectsFilename = '../trials_subNums'  # filename for trial-subject correspondence data
saved_pars_filename = '../weigths_lasg1.npz'    # filename for saved network parameters
depth_level = 0                         # Depth level for visualization (0, 1, 2)
sub_num = 1                             # Subject number to pick for test set

TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images, weights=None):
  """Build the EEG-CNN ConvNet model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  assign_ops = []
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 3, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 64])
  cnn_layer_shapes.append([3, 3, 64, 64])
  cnn_layer_shapes.append([3, 3, 64, 128])

  # Conv1_1
  layer_num = 0
  with tf.variable_scope('conv1_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    # kernel.assign(W_)
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_1)
    layer_num += 1

  # Conv1_2
  with tf.variable_scope('conv1_2',) as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_2)
    layer_num += 1

  # Conv1_3
  with tf.variable_scope('conv1_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(conv1_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv1_3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_3)
    layer_num += 1

  # Conv1_4
  with tf.variable_scope('conv1_4') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(conv1_3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv1_4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_4)
    layer_num += 1

  # pool1
  pool1 = tf.nn.max_pool(conv1_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')

  # Conv2_1
  with tf.variable_scope('conv2_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_3)
    layer_num += 1

  # Conv2_2
  with tf.variable_scope('conv2_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2_2)
    layer_num += 1

  # pool2
  pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

  # Conv3_1
  with tf.variable_scope('conv3_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(weights[layer_num * 2])
    assign_ops.append(tf.assign(kernel, W_))
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    bias = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3_1)

  # pool2
  pool3 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  return conv1_4, conv2_2, conv3_1, assign_ops


def inference_reverse_1(feature_map, weights=None, filt_num=0):
  """Build the EEG-CNN DeConvNet model.

  Args:
    feature_map: Selected feature map.

  Returns:
    reconstructed images.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  assign_ops = []
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 32, 3])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 1, 32])

  # We start from the last conv layer and backpropogate towards the input.

  # Conv1_4
  # The first last layer only looks at a single feature map and therefore only the corresponding filter is selected.
  layer_num = 3
  with tf.variable_scope('deconv1_4') as scope:
    rect_map = tf.nn.relu(feature_map, name=scope.name)
    biases = _variable_on_cpu('biases', [1,], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1][filt_num:filt_num+1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2][:, :, :, filt_num:filt_num+1], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Assign operations need to be fed separately to the tf.run() function in order to be run properly.
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    # deconv1_4 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_4 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_4)
    layer_num -= 1

  # Conv1_3
  with tf.variable_scope('deconv1_3') as scope:
    rect_map = tf.nn.relu(deconv1_4, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_3 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_3 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')


    _activation_summary(deconv1_3)
    layer_num -= 1

  # Conv1_2
  with tf.variable_scope('deconv1_2') as scope:
    rect_map = tf.nn.relu(deconv1_3, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_2 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_2 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_2)
    layer_num -= 1

  # Conv1_1
  with tf.variable_scope('deconv1_1') as scope:
    rect_map = tf.nn.relu(deconv1_2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_1 = tf.nn.deconv2d(unbiased_feature_map, kernel, [9, 32, 32, 3], strides=[1, 1, 1, 1], padding='SAME')
    deconv1_1 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_1)
    layer_num -= 1

  return deconv1_1, assign_ops

def inference_reverse_2(feature_map, weights=None, filt_num=0):
  """Build the EEG-CNN DeConvNet model.

  Args:
    feature_map: Selected feature map.

  Returns:
    reconstructed images.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  assign_ops = []
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 32, 3])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 64, 32])
  cnn_layer_shapes.append([3, 3, 1, 64])

  # We start from the last conv layer and backpropogate towards the input.

  # Conv2_2
  # The first last layer only looks at a single feature map and therefore only the corresponding filter is selected.
  layer_num = 5
  with tf.variable_scope('deconv2_2') as scope:
    rect_map = tf.nn.relu(feature_map, name=scope.name)
    biases = _variable_on_cpu('biases', [1,], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1][filt_num:filt_num+1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2][:, :, :, filt_num:filt_num+1], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    deconv2_2 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv2_2)
    layer_num -= 1

  # Conv2_1
  with tf.variable_scope('deconv2_1') as scope:
    rect_map = tf.nn.relu(deconv2_2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    deconv2_1 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')


    _activation_summary(deconv2_1)
    layer_num -= 1


  # Upsampling (Un-maxpool)
  unmaxpool1 = tf.image.resize_bicubic(deconv2_1, [32, 32])

  # Conv1_4
  # The first last layer only looks at a single feature map and therefore only the corresponding filter is selected.
  with tf.variable_scope('deconv1_4') as scope:
    rect_map = tf.nn.relu(unmaxpool1, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    # deconv1_4 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_4 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_4)
    layer_num -= 1

  # Conv1_3
  with tf.variable_scope('deconv1_3') as scope:
    rect_map = tf.nn.relu(deconv1_4, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_3 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_3 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')


    _activation_summary(deconv1_3)
    layer_num -= 1

  # Conv1_2
  with tf.variable_scope('deconv1_2') as scope:
    rect_map = tf.nn.relu(deconv1_3, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_2 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_2 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_2)
    layer_num -= 1

  # Conv1_1
  with tf.variable_scope('deconv1_1') as scope:
    rect_map = tf.nn.relu(deconv1_2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_1 = tf.nn.deconv2d(unbiased_feature_map, kernel, [9, 32, 32, 3], strides=[1, 1, 1, 1], padding='SAME')
    deconv1_1 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_1)
    layer_num -= 1

  return deconv1_1, assign_ops

def inference_reverse_3(feature_map, weights=None, filt_num=0):
  """Build the EEG-CNN DeConvNet model.

  Args:
    feature_map: Selected feature map.

  Returns:
    reconstructed images.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  assign_ops = []
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 32, 3])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 64, 32])
  cnn_layer_shapes.append([3, 3, 64, 64])
  cnn_layer_shapes.append([3, 3, 1, 64])

  # We start from the last conv layer and backpropogate towards the input.

  # Conv3_1
  layer_num = 6
  with tf.variable_scope('deconv3_1') as scope:
    rect_map = tf.nn.relu(feature_map, name=scope.name)
    biases = _variable_on_cpu('biases', [1,], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1][filt_num:filt_num+1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2][:, :, :, filt_num:filt_num+1], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    deconv3_1 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv3_1)
    layer_num -= 1


  # Upsampling (Un-maxpool)
  unmaxpool2 = tf.image.resize_bicubic(deconv3_1, [16, 16])

  # Conv2_2
  # The first last layer only looks at a single feature map and therefore only the corresponding filter is selected.
  with tf.variable_scope('deconv2_2') as scope:
    rect_map = tf.nn.relu(unmaxpool2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    deconv2_2 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv2_2)
    layer_num -= 1

  # Conv2_1
  with tf.variable_scope('deconv2_1') as scope:
    rect_map = tf.nn.relu(deconv2_2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    deconv2_1 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv2_1)
    layer_num -= 1


  # Upsampling (Un-maxpool)
  unmaxpool1 = tf.image.resize_bicubic(deconv2_1, [32, 32])

  # Conv1_4
  # The first last layer only looks at a single feature map and therefore only the corresponding filter is selected.
  with tf.variable_scope('deconv1_4') as scope:
    rect_map = tf.nn.relu(unmaxpool1, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    assign_ops.append(tf.assign(kernel, W_))
    # unbiased_feature_map.shape = [batch, height, width, in_channels]
    # kernel.shape = [height, width, output_channels, in_channels]
    # in_channels should be the same
    # deconv1_4 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_4 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_4)
    layer_num -= 1

  # Conv1_3
  with tf.variable_scope('deconv1_3') as scope:
    rect_map = tf.nn.relu(deconv1_4, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_3 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_3 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')


    _activation_summary(deconv1_3)
    layer_num -= 1

  # Conv1_2
  with tf.variable_scope('deconv1_2') as scope:
    rect_map = tf.nn.relu(deconv1_3, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_2 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')
    deconv1_2 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_2)
    layer_num -= 1

  # Conv1_1
  with tf.variable_scope('deconv1_1') as scope:
    rect_map = tf.nn.relu(deconv1_2, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][2], tf.constant_initializer(0.0))
    # Load baseline parameters from file
    b_ = tf.constant(weights[layer_num * 2 + 1])
    assign_ops.append(tf.assign(biases, b_))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    # Shape of kernel is selected to only include one filter
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    # Load weight matrix parameters from file
    W_ = tf.constant(np.transpose(weights[layer_num * 2], [0, 1, 3, 2]))
    assign_ops.append(tf.assign(kernel, W_))
    # deconv1_1 = tf.nn.deconv2d(unbiased_feature_map, kernel, [9, 32, 32, 3], strides=[1, 1, 1, 1], padding='SAME')
    deconv1_1 = tf.nn.conv2d(unbiased_feature_map, kernel, [1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_1)
    layer_num -= 1

  return deconv1_1, assign_ops

# Main loop
if __name__ == '__main__':
    # Load EEG images data from file.
    mat = scipy.io.loadmat(filename)
    featureMatrix = mat['featMat']
    labels = mat['labels']

    # Load trials subject information
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial
    ts = subjNumbers == sub_num
    tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))     # Training trials
    ts = np.squeeze(np.nonzero(ts))                     # Test trials

    # Load network parameters from file.
    saved_pars = np.load(saved_pars_filename)
    param_values = [saved_pars['arr_%d' % i] for i in range(len(saved_pars.files))]
    # Change the order of dimensions for parameters array to match that of TF
    for i in range(7):
        param_values[2 * i] = np.rollaxis(np.rollaxis(param_values[2*i], 0, 4), 0, 3)

    images = featureMatrix[0, tr, :]
    images = np.swapaxes(np.swapaxes(images, 1, 2), 2, 3).astype('float32')
    # Compute the feature maps after each pool layer
    [conv1, conv2, conv3, assign_ops] = inference(images, weights=param_values)

    # Initial values
    init = tf.initialize_all_variables()
    sess = tf.Session()
    # Initialize the graph
    sess.run(init)
    # Outputs array of feature maps [#samples, map_width, map_height, #filters]
    feature_maps = sess.run([conv1, conv2, conv3]+assign_ops)
    # pl.figure(); vis_square(np.rollaxis(feature_maps[0][0, :], 2, 0))
    # pl.figure(); vis_square(np.rollaxis(feature_maps[1][0, :], 2, 0))
    # pl.figure(); vis_square(np.rollaxis(feature_maps[2][0, :], 2, 0))

    deconv_funcs = {0: inference_reverse_1, 1: inference_reverse_2, 2:inference_reverse_3}
    # Loop over all filters
    for loop_counter, filt_num in enumerate(xrange(feature_maps[depth_level].shape[3])):
        select_feature_map = feature_maps[depth_level][:, :, :, filt_num:filt_num+1]
        feature_map_reshaped = select_feature_map.reshape((select_feature_map.shape[0], -1))
        feat_maps_max = np.mean(feature_map_reshaped, axis=1)
        best_indices = np.argsort(feat_maps_max)[-9:]

        # Plot the feature maps for selected images
        pl.figure();
        # pl.subplot(1,3,3); vis_square(np.swapaxes(np.squeeze(select_feature_map[best_indices]), 1, 2))
        pl.subplot(1,3,3); vis_square((np.squeeze(select_feature_map[best_indices])))
        # Reuse all variables for all iterations following the first one.
        if loop_counter > 0:
            tf.get_variable_scope().reuse_variables()

        [deconv1, assign_ops] = deconv_funcs[depth_level](select_feature_map[best_indices, :], weights=param_values, filt_num=filt_num)
        # [deconv1, assign_ops] = inference_reverse_1(select_feature_map, weights=param_values, filt_num=filt_num)

        # Initialized variables
        init = tf.initialize_all_variables()
        sess.run(init)

        reconst_image = sess.run([deconv1]+assign_ops)
        # pl.figure();
        pl.subplot(1,3,1); vis_square(np.swapaxes(images[best_indices], 1, 2)); #pl.grid(False, color='w')
        pl.subplot(1,3,2); vis_square(np.swapaxes(reconst_image[0], 1, 2)); #pl.grid(False, color='w')
        # pl.title('depth:{0}, filter#:{1}'.format(depth_level, filt_num))
        axes = pl.gcf().get_axes()
        for ax_num, ax in enumerate(axes):
            # axe.xaxis.set_visible(False)
            # axe.yaxis.set_visible(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if ax_num == 0:
              ax.xaxis.grid(True, color='k')
              ax.yaxis.grid(True, color='k')
              ax.set_xticks([select_feature_map[0].shape[1], 2*select_feature_map[0].shape[1]+1])
              ax.set_yticks([select_feature_map[0].shape[1], 2*select_feature_map[0].shape[1]+1])
            else:
              ax.xaxis.grid(True, color='w')
              ax.yaxis.grid(True, color='w')
              ax.set_xticks([reconst_image[0].shape[1], 2*reconst_image[0].shape[1]+1])
              ax.set_yticks([reconst_image[0].shape[1], 2*reconst_image[0].shape[1]+1])
        # pl.savefig('../Figures/Vis/vis_depth{0}_filter{1}'.format(depth_level, filt_num), dpi=300)
        pl.show()
    sess.close()
    print('Done!')

    # reconst = inference_reverse(pool1[0,:], weights=param_values)