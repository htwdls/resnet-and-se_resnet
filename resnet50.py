# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:52:44 2018

@author: LH
"""

import tensorflow as tf
LEARNINGRATE = 1e-3
def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)
def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)
def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')
def conv2d2(x, w):
    return tf.nn.conv2d(x, w, [1, 2, 2, 1], 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

#def _conv(name, x, filter_size, in_filters, out_filters, strides):
#    with tf.variable_scope(name):
#      n = filter_size * filter_size * out_filters
#      kernel = tf.get_variable(
#              'DW', 
#              [filter_size, filter_size, in_filters, out_filters],
#              tf.float32, 
#              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
#      return tf.nn.conv2d(x, kernel, strides, padding='SAME')



def _conv(net,keral_size,channel,filters):
    conv1_weights = tf.Variable(weight_variable([keral_size, keral_size, 
                    channel, filters], stddev=1e-4),name="conv1_weights")
    conv1_biases = tf.Variable(bias_variable([filters]), name="conv1_biases")
    conv1 = conv2d2(net, conv1_weights) + conv1_biases
    relu1 = tf.nn.relu(conv1)
    return relu1





def inference(features, one_hot_labels):
    number = 61
    print('features',features.shape)
    net = _conv(feature,3,3,64)
    #第一层
    