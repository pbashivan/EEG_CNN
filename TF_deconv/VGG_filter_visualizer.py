# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
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
import matplotlib.pyplot as pl
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

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


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
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_1)
    layer_num += 1

  # Conv1_2
  with tf.variable_scope('conv1_2',) as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_2)
    layer_num += 1

  # Conv1_3
  with tf.variable_scope('conv1_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(conv1_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1_3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_3)
    layer_num += 1

  # Conv1_4
  with tf.variable_scope('conv1_4') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(conv1_3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
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
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1_3)
    layer_num += 1

  # Conv2_2
  with tf.variable_scope('conv2_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
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
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3_1)

  # pool2
  pool3 = tf.nn.max_pool(conv3_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool3')

  return conv1_4, conv2_2, conv3_1

def inference_reverse(feature_map, weights=None, filt_num=0):
  """Build the EEG-CNN DeConvNet model.

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
  cnn_layer_shapes = [];
  cnn_layer_shapes.append([3, 3, 3, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 32])
  cnn_layer_shapes.append([3, 3, 32, 64])
  cnn_layer_shapes.append([3, 3, 64, 64])
  cnn_layer_shapes.append([3, 3, 64, 128])

  # Conv1_1
  layer_num = -1
  with tf.variable_scope('deconv1_4') as scope:
    rect_map = tf.nn.relu(feature_map, name=scope.name)
    biases = _variable_on_cpu('biases', cnn_layer_shapes[layer_num][3], tf.constant_initializer(0.0))
    unbiased_feature_map = tf.nn.bias_add(rect_map, -biases)
    kernel = _variable_with_weight_decay('weights', shape=cnn_layer_shapes[layer_num][:, :, :, filt_num],
                                         stddev=1e-4, wd=0.0)
    deconv1_4 = tf.nn.deconv2d(unbiased_feature_map, kernel, feature_map.shape, strides=[1, 1, 1, 1], padding='SAME')

    _activation_summary(deconv1_4)
    layer_num -= 1

  return deconv1_4


if __name__ == '__main__':
    # Load EEG images data from file.
    filename = '../EEG_images_32_timeWin'
    mat = scipy.io.loadmat(filename)
    featureMatrix = mat['featMat']
    labels = mat['labels']
    images = featureMatrix[0, 1:10, :]


    # Load network parameters from file.
    saved_pars_filename = '../weigths_lasg1.npz'
    saved_pars = np.load(saved_pars_filename)
    param_values = [saved_pars['arr_%d' % i] for i in range(len(saved_pars.files))]
    # Change the dimensions order of parameters array to match that of TF
    for i in range(7):
        param_values[2 * i] = np.rollaxis(np.rollaxis(param_values[2*i], 0, 4), 0, 3)

    images = np.swapaxes(np.swapaxes(images, 1, 2), 2, 3).astype('float32')
    # Compute the feature maps after each pool layer
    [pool1, pool2, pool3] = inference(images, weights=param_values)

    # Initial values
    init = tf.initialize_all_variables()
    sess = tf.Session()
    # Initialize the graph
    sess.run(init)
    # Outputs array of feature maps [#samples, map_width, map_height, #filters]
    feature_maps = sess.run([pool1, pool2, pool3])
    # pl.figure(); vis_square(np.rollaxis(filters[0][0, :], 2, 0))
    # pl.figure(); vis_square(np.rollaxis(filters[1][0, :], 2, 0))
    # pl.figure(); vis_square(np.rollaxis(filters[2][0, :], 2, 0))
    deconv1 = inference_reverse(feature_maps[0][0, :, :, 0], weights=param_values, filt_num=0)
    reconstructs = sess.run(deconv1)
    print('Done!')
    # reconst = inference_reverse(pool1[0,:], weights=param_values)