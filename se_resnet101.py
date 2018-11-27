# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:40:15 2018

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

def se_block(input_tensor,filters3,stage, block):
    
    gp_name_base = 'global_pool' + str(stage) + block + '_branch'
    fc_name_base = 'fc' + str(stage) + block + '_branch'
    sm_name_base = 'sigmodi' + str(stage) + block + '_branch'
    mul_name_base = 'mul' + str(stage) + block + '_branch' 
    pooled_inputs = tf.reduce_mean(input_tensor, [1, 2], name= gp_name_base + '2a_global_pool', keep_dims=True)
    
    down_inputs = tf.layers.conv2d(pooled_inputs, filters3 // 16, (1, 1), use_bias=True,
                                   name= fc_name_base + '_1x1_down', strides=(1, 1),
                                   padding='valid', 
                                   #data_format=data_format, 
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.zeros_initializer())
    
    down_inputs_relu = tf.nn.relu(down_inputs)
    up_inputs = tf.layers.conv2d(down_inputs_relu,filters3, (1, 1), use_bias=True,
                                 name= fc_name_base + '_1x1_up', strides=(1, 1),
                                 padding='valid', 
                                 #data_format=data_format, 
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 bias_initializer=tf.zeros_initializer())
    #print(up_inputs.get_shape())
    prob_outputs = tf.nn.sigmoid(up_inputs, name= sm_name_base + '_prob')
    rescaled_feat = tf.multiply(prob_outputs, input_tensor, name= mul_name_base + '_mul') 
    return rescaled_feat




def identity_block(input_tensor, kernel_size2, filters, stage, block):
    filters1, filters2, filters3 = filters
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = tf.layers.conv2d(inputs=input_tensor,filters=filters1,kernel_size=1,
            strides=1,padding='same',activation=None,name=conv_name_base+'2a')
    x = tf.layers.batch_normalization(x, axis=3,name=bn_name_base + '2a')
    x = tf.nn.relu(x)
    
    x = tf.layers.conv2d(inputs=x,filters=filters2,kernel_size=kernel_size2,
            strides=1,padding='same',activation=None,name=conv_name_base+'2b')
    x = tf.layers.batch_normalization(x, axis=3,name=bn_name_base + '2b')
    x = tf.nn.relu(x)
    
    x = tf.layers.conv2d(inputs=x,filters=filters3,kernel_size=1,
            strides=1,padding='same',activation=None,name=conv_name_base+'2c')
    x = tf.layers.batch_normalization(x, axis=3,name=bn_name_base + '2c')
    
    x = se_block(x,filters3,stage, block)
    
    x = tf.nn.relu(input_tensor + x)
    return x

def conv_block(input_tensor, kernel_size2, filters, stage, block, strides2=2):
    
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = tf.layers.conv2d(inputs=input_tensor,filters=filters1,kernel_size=1,
            strides=strides2,padding='same',activation=None,name=conv_name_base+'2a')
    x = tf.layers.batch_normalization(x, axis=3,name=bn_name_base + '2a')
    x = tf.nn.relu(x)
    
    x = tf.layers.conv2d(inputs=x,filters=filters2,kernel_size=kernel_size2,
            strides=1,padding='same',activation=None,name=conv_name_base+'2b')
    x = tf.layers.batch_normalization(x, axis=3,name=bn_name_base + '2b')
    x = tf.nn.relu(x)
    
    x = tf.layers.conv2d(inputs=x,filters=filters3,kernel_size=1,
            strides=1,padding='same',activation=None,name=conv_name_base+'2c')
    x = tf.layers.batch_normalization(x, axis=3,name=bn_name_base + '2c')
    x = se_block(x,filters3,stage, block)
    shortcut = tf.layers.conv2d(inputs=input_tensor,filters=filters3,kernel_size=1,
            strides=strides2,padding='same',activation=None,name=conv_name_base + '1')
    shortcut = tf.layers.batch_normalization(shortcut, axis=3,name=bn_name_base + '1')
    x = tf.nn.relu(shortcut + x)       
    return x

def inference(features, one_hot_labels):
    x = tf.layers.conv2d(inputs=features,filters=64,kernel_size=7,
            strides=2,padding='same',activation=None,name='conv1')
    x = tf.layers.batch_normalization(x, axis=3,name='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(inputs=x,pool_size=3,strides=2,
                      padding='same',name='maxpool1')
    #stage2#
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides2=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    #stage3#
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #stage4#
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r')    
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w')
    #stage5#
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    x = tf.layers.max_pooling2d(inputs=x,pool_size=7,strides=1,name='maxpool2')
    
    W_fc18 = weight_variable([1 * 1 * 2048, 1000])
    b_fc18 = bias_variable([1000])
    x_flat = tf.reshape(x, [-1, 1*1*2048])
    x = tf.nn.relu(tf.matmul(x_flat, W_fc18) + b_fc18)
    print('x',x.shape)
    keep_prob = tf.placeholder("float")
    W_fc2 = weight_variable([1000, 61])
    b_fc2 = bias_variable([61])
    prob = tf.matmul(x, W_fc2) + b_fc2
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=prob))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    return train_step, cross_entropy, prob, keep_prob