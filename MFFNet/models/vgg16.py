"""

VGG16

Author: Qiaoyi Yin
Date: 07.2019

"""


import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


class VGG16:

    def __init__(self, train_data, args_dict, scope):
        # Initialize the VGG16
        self.train_data = train_data
        self.args_dict = args_dict
        self.scope = scope

        # Allocate the values in the arguments dictionary
        self.bn = args_dict['bn']
        self.bn_decay = args_dict['bn_decay']
        self.num_output_channels = args_dict['num_output_channels']
        self.conv_ksize = args_dict['conv_ksize']
        self.pool_ksize = args_dict['pool_ksize']
        self.stride = args_dict['stride']
        self.image_size = args_dict['image_size']
        self.num_output_fc = args_dict['num_output_fc']
        self.train_flag = args_dict['train_flag']

    def model(self):
        """Function that is used to train the VGG16 network

        :param train_data: tensor, an object that stores the train data
        :param train_flag: bool, an object that represents whether the model is trainable
        :return:
        """

        with tf.variable_scope(self.scope) as scope:
            # 1st convlution layer
            self.conv1_1 = tf_util.conv2d(inputs=self.train_data, num_output_channels=self.num_output_channels[0],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv1_1',)
            # 2nd convlution layer
            self.conv1_2 = tf_util.conv2d(inputs=self.conv1_1, num_output_channels=self.num_output_channels[0],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv1_2')
            self.pool1 = tf_util.max_pool2d(inputs=self.conv1_2, kernel_size=self.pool_ksize, padding='SAME',
                                            stride=self.stride, scope='pool1')

            # 3rd convlution layer
            self.conv2_1 = tf_util.conv2d(inputs=self.pool1, num_output_channels=self.num_output_channels[1],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv2_1')
            # 4th convlution layer
            self.conv2_2 = tf_util.conv2d(inputs=self.conv2_1, num_output_channels=self.num_output_channels[1],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv2_2')
            self.pool2 = tf_util.max_pool2d(inputs=self.conv2_2, kernel_size=self.pool_ksize, padding='SAME',
                                            stride=self.stride, scope='pool2')

            # 5th convlution layer
            self.conv3_1 = tf_util.conv2d(inputs=self.pool2, num_output_channels=self.num_output_channels[2],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv3_1')
            # 6th convlution layer
            self.conv3_2 = tf_util.conv2d(inputs=self.conv3_1, num_output_channels=self.num_output_channels[2],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv3_2')
            # 7th convlution layer
            self.conv3_3 = tf_util.conv2d(inputs=self.conv3_2, num_output_channels=self.num_output_channels[2],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv3_3', )
            self.pool3 = tf_util.max_pool2d(inputs=self.conv3_3, kernel_size=self.pool_ksize, padding='SAME',
                                            stride=self.stride, scope='pool3')

            # 8th convlution layer
            self.conv4_1 = tf_util.conv2d(inputs=self.pool3, num_output_channels=self.num_output_channels[3],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv4_1')
            # 9th convlution layer
            self.conv4_2 = tf_util.conv2d(inputs=self.conv4_1, num_output_channels=self.num_output_channels[3],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv4_2')
            # 10th convlution layer
            self.conv4_3 = tf_util.conv2d(inputs=self.conv4_2, num_output_channels=self.num_output_channels[3],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv4_3')
            self.pool4 = tf_util.max_pool2d(inputs=self.conv4_3, kernel_size=self.pool_ksize, padding='SAME',
                                            stride=self.stride, scope='pool4')

            # 11th convlution layer
            self.conv5_1 = tf_util.conv2d(inputs=self.pool4, num_output_channels=self.num_output_channels[4],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv5_1')
            # 12th convlution layerb
            self.conv5_2 = tf_util.conv2d(inputs=self.conv5_1, num_output_channels=self.num_output_channels[4],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv5_2')
            # 13th convlution layer
            self.conv5_3 = tf_util.conv2d(inputs=self.conv5_3, num_output_channels=self.num_output_channels[4],
                                          kernel_size=self.conv_ksize, padding='SAME', stride=self.stride,
                                          bn=self.bn, bn_decay=self.bn_decay, is_training=self.train_flag, scope='conv5_3')
            self.pool5 = tf_util.max_pool2d(inputs=self.conv5_3, kernel_size=self.pool_ksize, padding='SAME',
                                            stride=self.stride, scope='pool5')

            # 14th affine layer / fully-connected layer
            x_flattened = tf.reshape(self.pool5, [-1, (self.image_size/32)*self.num_output_channels[4]])
            self.fc6 = tf_util.fully_connected(inputs=x_flattened, num_outputs=self.num_output_fc[0], use_xavier=True, scope='fc6',
                                               stddev=1e-3, activation_fn=tf.nn.relu, bn=self.bn, bn_decay=self.bn_decay,
                                               is_training=self.train_flag)
            self.fc6_drop = tf_util.dropout(inputs=self.fc6, is_training=self.train_flag, scope='fc6_drop',
                                            keep_prob=0.5, noise_shape=None)

            # 15th affine layer / fully-connected layer
            self.fc7 = tf_util.fully_connected(self.fc6_drop, num_outputs=self.num_output_fc[0], use_xavier=True, scope='fc7',
                                               stddev=1e-3, activation_fn=tf.nn.relu, bn=self.bn, bn_decay=self.bn_decay,
                                               is_training=self.train_flag)
            self.fc7_drop = tf_util.dropout(inputs=self.fc7, is_training=self.train_flag, scope='fc7_drop',
                                            keep_prob=0.5, noise_shape=None)

            # 16th affine layer / fully-connected layer
            self.fc8 = tf_util.fully_connected(inputs=self.fc7_drop, num_outputs=self.num_output_fc[1], use_xavier=True, scope='fc8',
                                               stddev=1e-3, activation_fn=tf.nn.relu, bn=self.bn, bn_decay=self.bn_decay,
                                               is_training=self.train_flag)

            return self.fc8


