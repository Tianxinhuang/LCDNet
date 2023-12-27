'''
Created on September 2, 2017

@author: optas
'''
import numpy as np
import tensorflow as tf
import random
from encoders_decoders import  conv2d,encoder_with_convs_and_symmetry, decoder_with_fc_only,decoder_with_folding_only
from tf_ops.sampling import tf_sampling
from pointnet_util import pointnet_sa_module_msg,pointnet_sa_module
def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        bnum=tf.shape(xyz)[0]
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        #new_xyz=tf_sampling.gather_point(xyz,idx)
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=np.arange(ptnum)
        ptids=tf.random_shuffle(ptids,seed=None)
        ptidsc=ptids[:npoint]
        ptid=tf.cast(tf.tile(tf.reshape(ptidsc,[-1,npoint,1]),[bnum,1,1]),tf.int32)
               
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
def global_fix(scope,cens,feats,mlp=[128,128],mlp1=[128,128]):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        tensor0=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
        #tensor0=tf.expand_dims(tf.concat([cens,feats,tf.tile(gfeat,[1,tf.shape(feats)[1],1])],axis=-1),axis=2)
        tensor=tensor0
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensorword=tensor
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
        tensor=tf.concat([tf.expand_dims(cens,axis=2),tf.expand_dims(feats,axis=2),tf.tile(tensor,[1,tf.shape(feats)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp1):
            tensor=conv2d('global_ptstate2%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensor=conv2d('global_ptout',tensor,3+feats.get_shape()[-1],[1,1],padding='VALID',activation_func=None)
        newcens=cens
        newfeats=feats+tf.squeeze(tensor[:,:,:,3:],[2])
    tf.add_to_collection('cenex',tf.reduce_mean(tf.abs(tensor[:,:,:,:3])))
    return newcens,newfeats
def local_kernel(l0_xyz,local=True,cenlist=None,pooling='max',it=True):
    l0_points=None
    if cenlist is None:
        cen11,cen22=None,None
    else:
        cen11,cen22=cenlist

    cen1,feat1=pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.2], [16], [[32,32,64]],cens=cen11, is_training=it, bn_decay=None, scope='layer1', use_nchw=False,bn=True,use_knn=True,pooling=pooling)
    cen2,feat2=pointnet_sa_module_msg(cen1, feat1, 128, [0.4], [16], [[64,64,128]],cens=cen22, is_training=it, bn_decay=None, scope='layer2',use_nchw=False,bn=True,use_knn=True,pooling=pooling)
    if not local:
        l3_xyz, rfeat3,_ = pointnet_sa_module(cen2, feat2, npoint=None, radius=None, nsample=None, mlp=[128,256,128], mlp2=None, group_all=True, is_training=it, bn_decay=None, scope='layer3',pooling=pooling)
        return [cen1,cen2],tf.squeeze(rfeat3,[1])
    else:
        cen3,feat3=pointnet_sa_module_msg(cen2, feat2, 32, [0.6], [16], [[128,128,256]], is_training=it, bn_decay=None, scope='layer3',use_nchw=False,bn=True,use_knn=True,pooling=pooling)
        rcen3,rfeat3=global_fix('global3',cen3,feat3,mlp=[256,256],mlp1=[256,256])
        return rcen3,rfeat3
def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, dnum=3, bneck_post_mlp=False,mode='fc'):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    #decoder = decoder_with_fc_only
    #decoder = decoder_with_folding_only

    n_input = [n_pc_points, dnum]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': False,
                    'non_linearity':tf.nn.relu
                    }
    if mode in ['ae','lae']:
        decoder = decoder_with_fc_only
        decoder_args = {'layer_sizes': [256,256, np.prod(n_input)],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False
                        }
    elif mode=='lfd':
        decoder = decoder_with_folding_only
        decoder_args = {'layer_sizes': [128,128,dnum],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False,
                        'local':True
                        }
    else:
        decoder = decoder_with_folding_only
        decoder_args = {'layer_sizes': [256,256,dnum],
                        'b_norm': False,
                        'b_norm_finish': False,
                        'verbose': False,
                        'local':False
                        }
    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args


from tf_ops.CD import tf_nndistance
def getidpts(pcd,ptid,ptnum):
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(pcd)[0],dtype=tf.int32),[-1,1,1]),[1,ptnum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    result=tf.gather_nd(pcd,idx)
    return result
def get_weight_variable(shape,stddev,name,regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
    #print(shape)
    weight = tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name+'/weights',weight)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight
def get_bias_variable(shape,value,name):
    bias = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    return bias
def conv2d(scope,inputs,num_outchannels,kernel_size,stride=[1,1],padding='SAME',stddev=1e-3,use_bnorm=False,activation_func=tf.nn.relu):
    with tf.variable_scope(scope):
        kernel_h,kernel_w=kernel_size
        num_inchannels=inputs.get_shape()[-1].value
        kernel_shape=[kernel_h,kernel_w,num_inchannels,num_outchannels]
        kernel=get_weight_variable(kernel_shape,stddev,'weights')
        stride_h,stride_w=stride
        outputs=tf.nn.conv2d(inputs,kernel,[1,stride_h,stride_w,1],padding=padding)
        bias = get_bias_variable([num_outchannels],0,'biases')
        outputs=tf.nn.bias_add(outputs,bias)
        if use_bnorm:
            outputs=tf.contrib.layers.batch_norm(outputs,
                                      center=True, scale=True,
                                      updates_collections=None,
                                      is_training=True,
                                      scope='bn')
        if activation_func!=None:
            outputs=activation_func(outputs)
    return outputs
def chamfer_wei(pcd1, pcd2, w1, w2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    pcd12=getidpts(pcd2,idx1,pcd1.get_shape()[1].value)
    pcd21=getidpts(pcd1,idx2,pcd2.get_shape()[1].value)
    dist1=tf.reduce_sum(tf.square(pcd1-pcd12),axis=-1)
    dist2=tf.reduce_sum(tf.square(pcd2-pcd21),axis=-1)
    dist1=tf.reduce_sum(w1*tf.sqrt(dist1),axis=1)
    dist2=tf.reduce_sum(w2*tf.sqrt(dist2),axis=1)
    dist=(tf.reduce_mean(dist1)+tf.reduce_mean(dist2))/2.0
    return dist
def F_Net(scope,data,mlp0=[64,64],mlp=[64,64,1],knum=8,inputfeat=None,feat=None):
    with tf.variable_scope(scope):
        result=tf.expand_dims(data,axis=2)
        for i, num_out_channel in enumerate([64,64]):
            result = conv2d('F_feats%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)
        feat=tf.reduce_max(result,axis=1,keepdims=True)

        result=tf.concat([tf.expand_dims(data,axis=2),tf.tile(feat,[1,tf.shape(data)[1],1,1]),tf.tile(inputfeat,[1,tf.shape(data)[1],1,1])],axis=-1)

        for i, num_out_channel in enumerate(mlp[:-1]):
            result = conv2d('F_trans%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.leaky_relu)

        result = conv2d('F_out%d'%i,result,mlp[-1],[1,1],activation_func=None,use_bnorm=False)
    result=tf.squeeze(result,[2])
    return result

def flow_feat(inputdata):
    result=tf.expand_dims(inputdata,axis=2)
    for i, num_out_channel in enumerate([64,128]):
        result = conv2d('F_feats%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)
    feat=tf.reduce_max(result,axis=1,keepdims=True)
    return feat

def sim_block(scope,data,args,feat=None,ifeat=None):
    with tf.variable_scope(scope):
        result=F_Net('module1',data,mlp=[64,16,1],inputfeat=feat,feat=ifeat)
        result=tf.squeeze(result,[2])
        result=-tf.square(result)
        result=args.sigma+tf.exp(result)
        result=result/(1e-8+tf.reduce_sum(result,axis=1,keepdims=True))
    return result

#To calculate loss
def lcdloss(pointcloud_pl,out,args):
    with tf.variable_scope('1ad'):
        igfeat=flow_feat(pointcloud_pl)
    with tf.variable_scope('1ad',reuse=True):
        iofeat=flow_feat(out)
    gfeat=tf.concat([igfeat,iofeat],axis=-1)
    with tf.variable_scope('2ad'):
        fi=sim_block('flow',pointcloud_pl,args,gfeat,igfeat)
    with tf.variable_scope('2ad',reuse=True):
        fr=sim_block('flow',out,args,gfeat,iofeat)

    loss_ri=chamfer_wei(pointcloud_pl,out,fi,fr)
    loss_d=-tf.log(loss_ri+args.sigma_r)
    loss_e=loss_ri
    return loss_e, loss_d
