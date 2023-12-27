#!/usr/bin/env python
# coding=utf-8
# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com

import tensorflow as tf
import ops
#from pointnet_util2 import pointnet_sa_module_msg
def pugan_dis(name,inputs,reuse = False,bn = False,start_number = 32,is_training=True):
    with tf.variable_scope(name, reuse=reuse):
        inputs = tf.expand_dims(inputs,axis=2)
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = ops.mlp_conv(inputs, [start_number, start_number * 2])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1],1, 1])], axis=-1)
            features = ops.attention_unit(features, is_training=is_training)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = ops.mlp_conv(features, [start_number * 4, start_number * 8])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            outputs = ops.mlp(features, [start_number * 8, 1])
            outputs = tf.reshape(outputs, [-1, 1])

    reuse = True
    #variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

    return outputs
def crn_dis(name,point_cloud, divide_ratio=2):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        l0_xyz = point_cloud
        l0_points = None
        num_point = point_cloud.get_shape()[1].value
        l1_xyz, l1_points = ops.pointnet_sa_module_msg(l0_xyz, l0_points, int(num_point/8), [0.1, 0.2, 0.4], [16, 32, 128],
                                                   [[32//divide_ratio, 32//divide_ratio, 64//divide_ratio], \
                                                    [64//divide_ratio, 64//divide_ratio, 128//divide_ratio], \
                                                    [64//divide_ratio, 96//divide_ratio, 128//divide_ratio]],
                                                   scope='layer1', use_nchw=False)
        patch_values=ops.mlp_conv2(l1_points, [1], bn=None, bn_params=None)
        return  patch_values
#lr0.0001, weight0.05
def pf_dis(name,inputs):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #outputs=ops.cmlp(inputs,bn=tf.contrib.layers.batch_norm)
        outputs=ops.cmlp(inputs,bn=None)
    return outputs
def d_loss(real,fake):
    #real = D(input_real)
    #fake = D(input_fake)
    real_loss = tf.reduce_mean(tf.square(real - 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))

    loss = (real_loss + fake_loss)/2

    return loss

def g_loss(fake):
    #fake = D(input_fake)

    fake_loss = tf.reduce_mean(tf.square(fake - 1.0))
    return fake_loss
#def crn_dis

