import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tensorflow as tf
import numpy as np
from models import vgg16
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module


vgg16_args_dict = {'num_output_channelsd': [64, 128, 256, 512, 512],
                   'conv_size': [3, 3],
                   'pool_size': [2, 2],
                   'stride': [1, 1],
                   'num_output_fc': [1024, 2048]}


def placeholder_inputs(batch_size, num_point, image_size, num_pf_views, num_rgb_views):
    rgb_projection_pl = tf.placeholder(tf.float32, shape=(num_rgb_views, batch_size, image_size, image_size, 3))
    pf_projection_pl = tf.placeholder(tf.float32, shape=(num_pf_views, batch_size, image_size, image_size, 1))
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 9))

    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))

    return pointclouds_pl, rgb_projection_pl, pf_projection_pl, labels_pl


def get_model(point_cloud, projection_data, is_training, image_size,
              num_class=13, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx9 (RGB XYZ NormalX NormalY NormalZ), output Bx(num_class) """

    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    # Slice and obtain RGB and point frequency projection data
    pf_data = []
    rgb_data = []
    for idx in range(projection_data[0].shape[0]):
        sliced_pf_data = tf.slice(projection_data[0], [idx, 0, 0, 0, 0], [1, -1, -1, -1, -1])
        sliced_pf_data = tf.reshape(sliced_pf_data, (batch_size, image_size, image_size, 1))
        pf_data.append(sliced_pf_data)

    for idx in range(projection_data[1].shape[0]):
        sliced_rgb_data = tf.slice(projection_data[1], [idx, 0, 0, 0, 0], [1, -1, -1, -1, -1])
        sliced_rgb_data = tf.reshape(sliced_rgb_data, (batch_size, image_size, image_size, 3))
        rgb_data.append(sliced_rgb_data)

    # Set arguments
    vgg16_args_dict['bn'] = True
    vgg16_args_dict['bn_decay'] = bn_decay
    vgg16_args_dict['image_size'] = image_size
    vgg16_args_dict['train_flag'] = is_training

    # MFFNet Structure
    with tf.variable_scope('mffnet') as scope:
        with tf.variable_scope('2d_module') as scope_2d:
            # VGG16 Module for Point Frequency Projection
            with tf.variable_scope('pf_module') as scope_pf:
                pf_net1 = vgg16.VGG16(pf_data[0], vgg16_args_dict, 'pf_net1').model()
                pf_net2 = vgg16.VGG16(pf_data[1], vgg16_args_dict, 'pf_net2').model()
                pf_net3 = vgg16.VGG16(pf_data[2], vgg16_args_dict, 'pf_net3').model()

            # VGG16 Module for RGB Projection
            with tf.variable_scope('rgb_module') as scope_rgb:
                rgb_net1 = vgg16.VGG16(rgb_data[0], vgg16_args_dict, 'rgb_net1').model()
                rgb_net2 = vgg16.VGG16(rgb_data[1], vgg16_args_dict, 'rgb_net2').model()
                rgb_net3 = vgg16.VGG16(rgb_data[2], vgg16_args_dict, 'rgb_net3').model()
                rgb_net4 = vgg16.VGG16(rgb_data[3], vgg16_args_dict, 'rgb_net4').model()
                rgb_net5 = vgg16.VGG16(rgb_data[4], vgg16_args_dict, 'rgb_net5').model()
                rgb_net6 = vgg16.VGG16(rgb_data[5], vgg16_args_dict, 'rgb_net6').model()

            net2d_inputs = tf.concat(values=[pf_net1, pf_net2, pf_net3, rgb_net1, rgb_net2, rgb_net3, rgb_net4, rgb_net5, rgb_net6],
                                     axis=1, name='net2d_concat')

            # Full connection layers in 2d-net module
            net2d_fc1 = tf_util.fully_connected(inputs=net2d_inputs, num_outputs=1024, use_xavier=True,
                                                scope='net2d_fc1', stddev=1e-3, activation_fn=tf.nn.relu, bn=True,
                                                bn_decay=bn_decay, is_training=is_training)
            net2dfc_drop1 = tf_util.dropout(inputs=net2d_fc1, is_training=is_training, scope='net2dfc_dp1',
                                            keep_prob=0.5, noise_shape=None)

            net2d_fc2 = tf_util.fully_connected(inputs=net2dfc_drop1, num_outputs=128, use_xavier=True,
                                                scope='net2d_fc2', stddev=1e-3, activation_fn=tf.nn.relu, bn=True,
                                                bn_decay=bn_decay, is_training=is_training)

        # PointNet++ Module
        with tf.variable_scope('pointnet2') as scope_pnet2:
            # Set Abstraction layers
            l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=64,
                                                               mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay, scope='pointnet2_sa_layer1')
            l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.2, nsample=64,
                                                               mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay, scope='pointnet2_sa_layer2')
            l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=64, radius=0.4, nsample=64,
                                                               mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                               is_training=is_training, bn_decay=bn_decay, scope='pointnet2_sa_layer3')
            l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None,
                                                               mlp=[256, 512, 1024], mlp2=None, group_all=True,
                                                               is_training=is_training, bn_decay=bn_decay, scope='pointnet2_sa_layer4')

            # Feature Propagation layers
            l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256, 256], is_training, bn_decay,
                                           scope='pointnet2_fp_layer1')
            l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256, 256], is_training, bn_decay,
                                           scope='pointnet2_fp_layer2')
            l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256, 128], is_training, bn_decay,
                                           scope='pointnet2_fp_layer3')
            l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128], is_training, bn_decay,
                                           scope='pointnet2_fp_layer4')

            # FC layers
            pointnet2 = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='pointnet2_fc1',
                                       bn_decay=bn_decay)
            end_points['feats'] = pointnet2

            # # Another network structure
            # # Set Abstraction layers
            # l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64,
            #                                                    mlp=[64,64,128], mlp2=None, group_all=False,
            #                                                    is_training=is_training, bn_decay=bn_decay,
            #                                                    scope='layer1')
            # l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64,
            #                                                    mlp=[128,128,256], mlp2=None, group_all=False,
            #                                                    is_training=is_training, bn_decay=bn_decay,
            #                                                    scope='layer2')
            # l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
            #                                                    mlp=[256,512,1024], mlp2=None, group_all=True,
            #                                                    is_training=is_training, bn_decay=bn_decay,
            #                                                    scope='layer3')
            #
            # # Feature Propagation layers
            # l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay,
            #                                scope='fa_layer1')
            # l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay,
            #                                scope='fa_layer2')
            # l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([l0_xyz,l0_points],axis=-1), l1_points,
            #                                [128,128,128], is_training, bn_decay, scope='fa_layer3')
            #
            # # FC layers
            # net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1',
            #                      bn_decay=bn_decay)
            # end_points['feats'] = net
            # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
            # net = tf_util.conv1d(net, 13, 1, padding='VALID', activation_fn=None, scope='fc2')

        # MFFNet full connection layer
        mffnet_inputs = tf.concat(values=[pointnet2, net2d_fc2], axis=1, name='mffnet_concat')
        mffnet_fc1 = tf_util.conv1d(mffnet_inputs, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='mffnet_fc1',
                                    bn_decay=bn_decay)

        mffnet_drop1 = tf_util.dropout(mffnet_fc1, keep_prob=0.5, is_training=is_training, scope='mffnet_dp1')
        mffnet = tf_util.fully_connected(inputs=mffnet_drop1, num_outputs=num_class, use_xavier=True,
                                         scope='mffnet_fc2', stddev=1e-3, activation_fn=tf.nn.relu, bn=True,
                                         bn_decay=bn_decay, is_training=is_training)

        return mffnet


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
