#!/usr/bin/env python
# coding=utf-8
# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import numpy as np
import os
import sys
#sys.path.append('tf_ops/sampling')
#sys.path.append('tf_ops/grouping')
#sys.path.append('/htxnet/deconv')
from tf_ops.sampling import tf_sampling
from tf_ops.grouping import tf_grouping
#from tf_sampling import farthest_point_sample, gather_point
#from tf_grouping import query_ball_point, group_point, knn_point
#sys.path.append(os.path.dirname(os.getcwd()))
def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

##################################################################################
# Back projection Blocks
##################################################################################
def PointShuffler(inputs, scale=2):
    #inputs: B x N x 1 X C
    #outputs: B x N*scale x 1 x C//scale
    outputs = tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1,tf.shape(inputs)[3]//scale,scale])
    outputs = tf.transpose(outputs,[0, 1, 4, 3, 2])

    outputs = tf.reshape(outputs,[tf.shape(inputs)[0],tf.shape(inputs)[1]*scale,1,tf.shape(inputs)[3]//scale])

    return outputs

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, scope, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        p1_idx = tf_sampling.farthest_point_sample(npoint, xyz)
        new_xyz = tf_sampling.gather_point(xyz, p1_idx)
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = tf_grouping.query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = tf_grouping.group_point(xyz, idx) #b*n*k*3
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = tf_grouping.group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = conv2d(grouped_points, num_out_channel, [1,1],weight_decay=0.000001,
                                        padding='VALID', stride=[1,1], scope='conv%d_%d'%(i,j),activation_fn=tf.nn.leaky_relu)
            if use_nchw: grouped_points = tf.transpose(grouped_points, [0,2,3,1])
            new_points = tf.reduce_max(grouped_points, axis=[2]) #b*n*c
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat
def cmlp(inputs,mlps=[64,64,128,256],fcs=[256,128,16],bn=None, bn_params=None):
    x = tf.contrib.layers.conv1d(
                inputs, mlps[0],
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_0')
    x64=tf.contrib.layers.conv1d(
                x, mlps[1],
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_1')
    x128=tf.contrib.layers.conv1d(
                x64, mlps[2],
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_2')
    x256=tf.contrib.layers.conv1d(
                x128, mlps[3],
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_3')
    x64=tf.reduce_max(x64,axis=1)
    x128=tf.reduce_max(x128,axis=1)
    x256=tf.reduce_max(x256,axis=1)
    xs=tf.concat([x64,x128,x256],axis=-1)
    for i in range(len(fcs)):
        xs = tf.contrib.layers.fully_connected(
               xs, fcs[i],
               normalizer_fn=bn,
               normalizer_params=bn_params,
               activation_fn=tf.nn.relu,
               scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        xs, 1,
        activation_fn=None,
        scope='fc_out')
    return outputs


def mlp_conv2(inputs, layer_dims, bn=None, bn_params=None,name=''):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i, num_out_channel in enumerate(layer_dims[:-1]):
            inputs = tf.contrib.layers.conv1d(
                inputs, num_out_channel,
                kernel_size=1,
                normalizer_fn=bn,
                normalizer_params=bn_params,
                scope='conv_%d' % i)
        outputs = tf.contrib.layers.conv1d(
            inputs, layer_dims[-1],
            kernel_size=1,
            activation_fn=None,
            scope='conv_%d' % (len(layer_dims) - 1))
        return outputs

def coordinate_reconstruction_unit(inputs,scope="reconstruction",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        coord = conv2d(inputs, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)
        outputs = tf.squeeze(coord, [2])

        return outputs


def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)


        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x


##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0], tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov



def add_scalar_summary(name, value,collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])
def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])

def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])

def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update

