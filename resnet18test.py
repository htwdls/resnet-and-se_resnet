# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:06:20 2018

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

def inference(features, one_hot_labels):
    number = 80
    print('features',features.shape)
    W_conv1 = weight_variable([3, 3, 3, 64], stddev=1e-4)
    b_conv1 = bias_variable([64])
    net1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1)
    #net = max_pool_3x3(net)
    '''
    net = tf.layers.max_pooling2d(inputs=net,pool_size=3,strides=2,
                                  padding='same',name='maxpool0')
    '''
    #第一层
    #shortcut = net0
    W_conv2 = weight_variable([3, 3, 64, 64], stddev=1e-4)
    b_conv2 = bias_variable([64])
    net2 = tf.nn.relu(conv2d(net1, W_conv2) + b_conv2)
    W_conv3 = weight_variable([3, 3, 64, 64], stddev=1e-4)
    b_conv3 = bias_variable([64])
    net3 = (conv2d(net2, W_conv3) + b_conv3)
    #net = tf.nn.relu(shortcut + f)
    net3 = tf.nn.relu(net3)
    #shortcut = net
    W_conv4 = weight_variable([3, 3, 64, 64], stddev=1e-4)
    b_conv4 = bias_variable([64])
    net4 = tf.nn.relu(conv2d(net3, W_conv4) + b_conv4)
    W_conv5 = weight_variable([3, 3, 64, 64], stddev=1e-4)
    b_conv5 = bias_variable([64])
    net5 = (conv2d(net4, W_conv5) + b_conv5)
    #net = tf.nn.relu(shortcut + f)
    net5 = tf.nn.relu(net5)
     #第二层
    #shortcut = net
    W_conv6 = weight_variable([3, 3, 64, 128], stddev=1e-4)
    b_conv6 = bias_variable([128])
    net6 = tf.nn.relu(conv2d2(net5, W_conv6) + b_conv6)
    W_conv7 = weight_variable([3, 3, 128, 128], stddev=1e-4)
    b_conv7 = bias_variable([128])
    net7 = (conv2d(net6, W_conv7) + b_conv7)
    #net = tf.nn.relu(shortcut + f)
    net7 = tf.nn.relu(net7)
    #shortcut = net
    W_conv8 = weight_variable([3, 3, 128, 128], stddev=1e-4)
    b_conv8 = bias_variable([128])
    net8 = tf.nn.relu(conv2d(net7, W_conv8) + b_conv8)
    W_conv9 = weight_variable([3, 3, 128, 128], stddev=1e-4)
    b_conv9 = bias_variable([128])
    net9 = (conv2d(net8, W_conv9) + b_conv9)
    #net = tf.nn.relu(shortcut + f)
    net9 = tf.nn.relu(net9)
    
    #第三层
    #shortcut = net
    W_conv10 = weight_variable([3, 3, 128, 256], stddev=1e-4)
    b_conv10 = bias_variable([256])
    net10 = tf.nn.relu(conv2d2(net9, W_conv10) + b_conv10)
    W_conv11 = weight_variable([3, 3, 256, 256], stddev=1e-4)
    b_conv11 = bias_variable([256])
    net11 = (conv2d(net10, W_conv11) + b_conv11)
    net11 = tf.nn.relu(net11)
    #shortcut = net
    W_conv12 = weight_variable([3, 3, 256, 256], stddev=1e-4)
    b_conv12 = bias_variable([256])
    net12 = tf.nn.relu(conv2d(net11, W_conv12) + b_conv12)
    W_conv13 = weight_variable([3, 3, 256, 256], stddev=1e-4)
    b_conv13 = bias_variable([256])
    net13 = (conv2d(net12, W_conv13) + b_conv13)
    net13 = tf.nn.relu(net13)
    
    #第四层
    #shortcut = net
    W_conv14 = weight_variable([3, 3, 256, 512], stddev=1e-4)
    b_conv14 = bias_variable([512])
    net14 = tf.nn.relu(conv2d2(net13, W_conv14) + b_conv14)
    W_conv15 = weight_variable([3, 3, 512, 512], stddev=1e-4)
    b_conv15 = bias_variable([512])
    net15 = (conv2d(net14, W_conv15) + b_conv15)
    net15 = tf.nn.relu(net15)
    #shortcut = net
    W_conv16 = weight_variable([3, 3, 512, 512], stddev=1e-4)
    b_conv16 = bias_variable([512])
    net16 = tf.nn.relu(conv2d(net15, W_conv16) + b_conv16)
    W_conv17 = weight_variable([3, 3, 512, 512], stddev=1e-4)
    b_conv17 = bias_variable([512])
    net17 = (conv2d(net16, W_conv17) + b_conv17)
    #net = tf.nn.relu(shortcut + f)
    net17 = tf.nn.relu(net17)
    
    #输出层
    W_fc18 = weight_variable([16 * 16 * 512, 1000])
    b_fc18 = bias_variable([1000])
    net_flat = tf.reshape(net17, [-1, 16*16*512])
    h_fc1 = tf.nn.relu(tf.matmul(net_flat, W_fc18) + b_fc18)
    print('h_fc1',h_fc1.shape)
    # introduce dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # fc2
    W_fc2 = weight_variable([1000, 80])
    b_fc2 = bias_variable([80])
    prob = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    
    '''
    keep_prob = tf.placeholder("float")
    #h_pool3_flat = tf.reshape(h_pool3, [-1, 16*16*64])
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    net = tf.layers.average_pooling2d(inputs=net,pool_size=4,strides=1,
                                      name='avgpool1')
    net = tf.layers.flatten(inputs=net)
    W_fc18 = weight_variable([512, 80])
    b_fc18 = bias_variable([80])
    net = tf.matmul(net, W_fc18) + b_fc18
    print(net.shape)
    logits = tf.layers.dense(inputs=net,units=number,
                             activation=tf.nn.relu,name='logits')
    prob = tf.nn.softmax(net, name='prob')
    '''
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=prob))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    return train_step, cross_entropy, prob, keep_prob