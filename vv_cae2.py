import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
import tf_util
import copy
import random
#import point_choose
DATA_DIR=getdata.getspdir()
#import open3d as o3d
#from sklearn.cluster import KMeans
#from sklearn.mixture import GaussianMixture

filelist=os.listdir(DATA_DIR)

from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from ae_templates import mlp_architecture_ala_iclr_18,adaptive_loss_net,local_kernel,sampling,get_topk,find_diff,global_kernel,movenet,clossnet
from tf_ops.grouping import tf_grouping
#query_ball_point, group_point
from provider import shuffle_points, shuffle_data,shuffle_points,rotate_point_cloud,jitter_point_cloud
from dgcnn import dgcnn_kernel,dgcls_kernel
from discriminator import pugan_dis, crn_dis,pf_dis, g_loss, d_loss

trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))
#testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

EPOCH_ITER_TIME=2000
BATCH_ITER_TIME=5000
BASE_LEARNING_RATE=0.01
REGULARIZATION_RATE=0.0001
BATCH_SIZE=16
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7
PT_NUM=2048
FILE_NUM=6
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from data_util import lmdb_dataflow
tf.set_random_seed(1)

def get_gen(path,batch_size=BATCH_SIZE,input_ptnum=2048,output_ptnum=PT_NUM,is_training=True):
    df, num = lmdb_dataflow(path,batch_size,input_ptnum,output_ptnum,is_training)
    gen = df.get_data()
    return gen,num
def grouping(xyz,new_xyz, radius, nsample, points, knn=False, use_xyz=True):
    if knn:
        _,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
        #print('idx',idx)
        #assert False
    else:
        _,id0 = tf_grouping.knn_point(1, xyz, new_xyz)
        valdist,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
        idx=tf.where(tf.greater(valdist,radius),tf.tile(id0,[1,1,nsample]),idx)
        #print(valdist,idx,id0)
        #assert False

        #idx, pts_cnt = tf_grouping.query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = tf_grouping.group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = tf_grouping.group_point(points, idx) # (batch_size, npoint, nsample, channel)
        #print(grouped_points)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
        grouped_points=grouped_xyz

    return grouped_xyz,grouped_points
def chamfer_func(input,output):
    dis_i=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.tile(tf.expand_dims(output,axis=1),multiples=[1,tf.shape(input)[1],1,1])),axis=-1))
    dis_o=tf.transpose(dis_i,[0,2,1])
    dis_ii=tf.reduce_mean(tf.reduce_min(dis_i,axis=-1))
    dis_oo=tf.reduce_mean(tf.reduce_min(dis_o,axis=-1))
    dis=tf.reduce_mean(tf.maximum(dis_ii,dis_oo))
    return dis
#def chamfer_big(pcd1, pcd2):
#    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
#    dist1 = tf.reduce_mean(tf.sqrt(dist1))
#    dist2 = tf.reduce_mean(tf.sqrt(dist2))
#    return (dist1 + dist2) / 2
def getidpts(pcd,ptid,ptnum):
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(pcd)[0],dtype=tf.int32),[-1,1,1]),[1,ptnum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    result=tf.gather_nd(pcd,idx)
    return result
def cweight(dist,idx,length,frac,lamda):
    #print()
    countlist=[]
    for i in range(BATCH_SIZE):
        count=tf.bincount(idx[i],minlength=length)
        countlist.append(tf.expand_dims(count,axis=0))
    count=tf.concat(countlist,axis=0)
    count=tf.expand_dims(count,axis=-1)
    weight=tf.squeeze(tf.cast(getidpts(count,idx,length),tf.float32),[-1])
    #print('>>>>',weight)

    #weight=tf.pow(weight,lamda)
    #print(weight)
    weight=tf.sqrt(weight)
    weight=frac/(weight+1e-6)
    return weight
def chamfer_dcd(pcd1,pcd2,alpha=200,lamda=0.5):
    ptnum1=pcd1.get_shape()[1].value
    ptnum2=pcd2.get_shape()[1].value
    frac12=ptnum1/ptnum2
    frac21=ptnum2/ptnum1
    ptnum=np.maximum(ptnum1,ptnum2)
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    expdist1,expdist2=tf.exp(-dist1*alpha),tf.exp(-dist2*alpha)

    weight1=cweight(dist1,idx1,ptnum,frac21,lamda)
    weight2=cweight(dist2,idx2,ptnum,frac12,lamda)

    loss1=tf.reduce_mean(1.0-weight1*expdist1,axis=1)
    loss2=tf.reduce_mean(1.0-weight2*expdist2,axis=1)
    loss=tf.reduce_mean((loss1+loss2)/2)
    return loss
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    pcd12=getidpts(pcd2,idx1,pcd1.get_shape()[1].value)
    pcd21=getidpts(pcd1,idx2,pcd2.get_shape()[1].value)
    dist1=tf.sqrt(tf.reduce_sum(tf.square(pcd1-pcd12),axis=-1))
    dist2=tf.sqrt(tf.reduce_sum(tf.square(pcd2-pcd21),axis=-1))
    #dist1=tf.sqrt(dist1)
    #dist2=tf.sqrt(dist2)
    dist=(tf.reduce_mean(dist1)+tf.reduce_mean(dist2))/2.0
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return dist#tf.reduce_mean((dist1 + dist2) / 2)
    #return (dist1 + dist2) / 2
def emd_func(pred,gt):
    print('in')
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    dist = tf.reduce_mean(dist,axis=-1)
    
    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    #dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    #print(matched_out,dist)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist)
    return emd_loss
def chamfer_max(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #dist=tf.reduce_mean(tf.maximum(tf.sqrt(dist1),tf.sqrt(dist2)))
    #dist=(dist1 + dist2)/2
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    dist1=tf.reduce_mean(dist1)
    dist2=tf.reduce_mean(dist2)
    dist=tf.maximum(dist1, dist2)
    return dist,idx1

def chamfer(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    #dist1 = tf.reduce_mean(tf.sqrt(dist1))
    #dist2 = tf.reduce_mean(tf.sqrt(dist2))
    #dist=tf.reduce_mean(tf.maximum(tf.sqrt(dist1),tf.sqrt(dist2)))
    #dist=(dist1 + dist2)/2
    dist1 = tf.reduce_mean(dist1)
    dist2 = tf.reduce_mean(dist2)
    dist=(dist1 + dist2)
    #dist=tf.maximum(dist1,dist2)
    return dist,idx1
def chamfer_local(pcda,pcdb):
    ptnum=pcda.get_shape()[1].value
    knum=pcda.get_shape()[2].value
    pcd1=tf.reshape(pcda,[-1,knum,3])
    pcd2=tf.reshape(pcdb,[-1,knum,3])
    #pcd1,pcd2=normalize(pcd1,pcd2)
    #pcd1,pcd2=normalize(pcd1),normalize(pcd2)

    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    #dist1,dist2=tf.sqrt(dist1),tf.sqrt(dist2)
    dist1=tf.reduce_mean(tf.reshape(dist1,[-1,ptnum,knum]),axis=-1)#+0.1*tf.reduce_max(tf.reshape(dist1,[-1,ptnum,knum]),axis=1)#batch*k
    dist2=tf.reduce_mean(tf.reshape(dist2,[-1,ptnum,knum]),axis=-1)#+0.1*tf.reduce_max(tf.reshape(dist2,[-1,ptnum,knum]),axis=1)#batch*k
    #dist1,dist2=tf.sqrt(dist1),tf.sqrt(dist2)

    #dist=tf.reduce_mean(tf.maximum(tf.reduce_mean(dist1,axis=1)+tf.reduce_max(dist1,axis=1),tf.reduce_mean(dist2,axis=1)+tf.reduce_max(dist2,axis=1)))#(batch,)
    #dist=tf.reduce_mean(tf.maximum(tf.reduce_mean(dist1,axis=1),tf.reduce_mean(dist2,axis=1)))#(batch,)
    #dist=tf.reduce_mean((tf.reduce_mean(dist1,axis=1)+tf.reduce_mean(dist2,axis=1)))#(batch,)
    #dist=tf.reduce_mean(tf.maximum(tf.reduce_mean(dist1,axis=1),tf.reduce_mean(dist2,axis=1)))#(batch,)
    dist=tf.reduce_mean(tf.maximum(dist1,dist2))
    #dist=tf.reduce_mean(tf.reduce_max(tf.maximum(dist1,dist2),axis=-1))
    #dist=tf.reduce_mean((dist1+dist2))
    return dist,idx1
def multi_chamfer(n,inputpts,output,k=64,r=0.01,use_knn=True):
    in_cen=sampling(n,inputpts,use_type='f')[-1]
    out_cen=sampling(n,output,use_type='f')[-1]
    #in_cen=tf.concat([in_cen,out_cen],axis=1)

    out_kneighbor,_=grouping(output,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=use_knn, use_xyz=True)
    in_kneighbor,_=grouping(inputpts,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=use_knn, use_xyz=True)
    local_in=chamfer_local(in_kneighbor,out_kneighbor)[0]

    #out_kneighbor,_=grouping(output,new_xyz=out_cen, radius=r, nsample=k, points=None, knn=use_knn, use_xyz=True)
    #in_kneighbor,_=grouping(inputpts,new_xyz=out_cen, radius=r, nsample=k, points=None, knn=use_knn, use_xyz=True)
    #local_out=chamfer_local(in_kneighbor,out_kneighbor)[0]

    local_loss=local_in#(local_in+local_out)/2
    #local_loss=tf.maximum(local_in,local_out)
    return local_loss
def multi_chamfer_func(n,inputpts,output,klist=[16,32,64]):
    #result=[chamfer(inputpts,output)[0]]
    result=[]
    for k in klist:
        result.append(multi_chamfer(n,inputpts,output,k))
    result=tf.add_n(result)/len(klist)
    result=(result+0.1*chamfer_max(inputpts,output)[0])
    return result

def relative_loss(words):
    #vecs=tf.squeeze(words,[2])
    vecs=words
    vecs_trans=tf.transpose(vecs,[0,2,1])
    #rela_mat=tf.matmul(vecs,vecs_trans)
    dismat=tf.sqrt(tf.reduce_sum(tf.square(vecs),axis=-1,keepdims=True))

    rela_mat=tf.matmul(vecs,vecs_trans)/(dismat*tf.transpose(dismat,[0,2,1])+1e-5)-tf.expand_dims(tf.eye(words.get_shape()[1].value),axis=0)#batch*2048*2048
    relative_loss=tf.reduce_max(tf.reduce_max(rela_mat,axis=-1),axis=-1)
    relative_loss=tf.reduce_mean(relative_loss)
    return relative_loss
def pt_dev(pointcloud,partnum=4):
     bnum=tf.shape(xyz)[0]
     ptnum=xyz.get_shape()[1].value
     ptids=arange(ptnum)
     random.shuffle(ptids)
     ptid=tf.tile(tf.constant(ptids,shape=[1,ptnum,1],dtype=tf.int32),[bnum,1,1])
     bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,ptnum,1])
     idx=tf.concat([bid,ptid],axis=-1)
     new_xyz=tf.gather_nd(xyz,idx)
     result=[]
     devnum=ptnum//partnum
     for i in range(partnum):
         result.append(new_xyz[:,i*devnum:(i+1)*devnum,:])
     return result
def ptloss(ptin,ptout):
    with tf.variable_scope('ad'):
        inputword,meanin,varin=adaptive_loss_net(ptin)
    with tf.variable_scope('ad',reuse=True):
        outputword,meanout,varout=adaptive_loss_net(ptout)
    dzhengze=tf.reduce_mean(tf.reduce_sum(inputword,axis=-1))
    loss_e=ge_loss(inputword,outputword)+0.025*ge_loss(varin,varout)+0.025*ge_loss(meanin,meanout)
    loss_d=ad_loss(inputword,outputword)+0.025*ad_loss(varin,varout)+0.025*ad_loss(meanin,meanout)+0.003*dzhengze
    return loss_e,loss_d
def dev_loss(pointcloud,outcloud,partnum=4):
    with tf.variable_scope('ad'):
        inputword,meanin,varin=adaptive_loss_net(ptin)
    with tf.variable_scope('ad',reuse=True):
        outputword,meanout,varout=adaptive_loss_net(ptout)
    dzhengze=tf.reduce_mean(tf.reduce_sum(inputword,axis=-1))
    #argin=tf.
def exclusion_loss(ptin,ptout,rank=5):
    losslist=[]
    inlist=[]
    outlist=[]
    meanlist=[]
    varlist=[]
    with tf.variable_scope('ad'):
        for i in range(rank):
            inputword,meanin,varin=adaptive_loss_net(ptin,n_filter=[64,128,128])
            inlist.append(tf.expand_dims(inputword,1))
            meanlist.append(tf.expand_dims(meanin,1))
            varlist.append(tf.expand_dims(varin,1))
    inmat=tf.concat(inlist,axis=1)
    meanmat=tf.concat(meanlist,axis=1)
    varmat=tf.concat(varlist,axis=1)
    meanlist=[]
    varlist=[]
    with tf.variable_scope('ad',reuse=True):
        for i in range(rank):
            outputword,meanout,varout=adaptive_loss_net(ptout,n_filter=[64,128,128])
            outlist.append(tf.expand_dims(outputword,1))
            meanlist.append(tf.expand_dims(meanout,1))
            varlist.append(tf.expand_dims(varout,1))
    outmat=tf.concat(outlist,axis=1)#batch*rank*len
    meanoutmat=tf.concat(meanlist,axis=1)
    varoutmat=tf.concat(varlist,axis=1)
    meanloss=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(meanmat-meanoutmat),axis=-1),axis=-1))
    varloss=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(varmat-varoutmat),axis=-1),axis=-1))
    basic_loss_e=tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(tf.square(inmat-outmat),axis=-1),axis=-1))
    basic_loss_d=-basic_loss_e
    rank_mat=tf.reduce_sum(tf.square(tf.expand_dims(inmat,axis=2)-tf.expand_dims(inmat,axis=1)),axis=-1)#batch*rank*rank
    rank_loss=-tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(rank_mat,axis=-1)/(rank-1),axis=-1))
    dzhengze=tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(inmat,axis=1),axis=-1))
    loss_e=basic_loss_e+0.5*meanloss+0.5*varloss
    loss_d=basic_loss_d+0.05*rank_loss+0.002*dzhengze
    return loss_e,loss_d,dzhengze
def KL_divergence(words):
    meanword,varword=tf.nn.moments(words,[1])
    KL=tf.reduce_sum((tf.square(meanword)+varword-tf.log(varword+1e-5)-1)/2,axis=-1)
    KL=tf.reduce_mean(KL)
    return KL
#def KL_divergence(wordin,wordout):
def kf_loss(wordsin,wordsout):
    new_wordsin,meanwordin,stdwordin=normal_process(wordsin)
    new_wordsout,meanwordout,stdwordout=normal_process(wordsout)
    kf=tf.reduce_mean(tf.reduce_mean(tf.abs(tf.reduce_sum(tf.square(new_wordsin),axis=1)-tf.reduce_sum(tf.square(new_wordsout),axis=1)),axis=-1))
    return kf
#batch*2048*n
def sort_feat(words):
    ptnum=words.get_shape()[1].value
    featnum=words.get_shape()[-1].value
    trans_word=tf.transpose(words,[0,2,1])#batch*128*2048
    valuemat,kindex=tf.nn.top_k(trans_word,ptnum)
    return tf.transpose(valuemat,[0,2,1])#batch*2048*128
def ks_loss(sortedin,sortedout):
    dismat=tf.square(sortedin-sortedout)
    ks=tf.reduce_max(dismat,axis=1)
    result=tf.reduce_mean(ks,[0,1])
    return result
def mmd_loss(wordsin,wordsout):
    ptnum=wordsin.get_shape()[1].value
    X=tf.expand_dims(wordsin,axis=2)
    Y=tf.expand_dims(wordsout,axis=2)
    ksrc=tf.reduce_mean(X*tf.transpose(X,[0,2,1,3]),axis=[1,2])
    ktar=tf.reduce_mean(Y*tf.transpose(Y,[0,2,1,3]),axis=[1,2])
    kmix=tf.reduce_mean(X*tf.transpose(Y,[0,2,1,3]),axis=[1,2])
    mmd=ksrc+ktar-2*kmix
    return mmd
def topk_feat(words,k):
    trans_word=tf.transpose(words,[0,2,1])#batch*128*2048
    valuemat,kindex=tf.nn.top_k(trans_word,k)
    return tf.transpose(valuemat,[0,2,1])#batch*k*128
def topk_ge(wordin,wordout,k=10):
    kin=topk_feat(wordin,k)
    kout=topk_feat(wordout,k)
    #lengthin=tf.sqrt(tf.reduce_sum(tf.square(kin),keepdims=True,axis=-1))
    #lengthout=tf.sqrt(tf.reduce_sum(tf.square(kout),keepdims=True,axis=-1))
    result=tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.square(kin-kout),axis=-1),axis=1))
    return result
#batch*2048*128
def topk_nearfeats(words,k=16):
    trans_word=tf.transpose(words,[0,2,1])#batch*128*2048
    _,kidx=tf.nn.top_k(trans_word,k)#batch*128*k
    #trans_idx=tf.transpose(kindex,[0,2,1])#batch*k*128
    bidx=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(words)[0],dtype=tf.int32),[-1,1,1,1]),[1,tf.shape(words)[-1],k,1])
    idx=tf.concat([bidx,tf.expand_dims(kidx,axis=-1)],axis=-1)
    kwords=tf.gather_nd(words,idx)#batch*128*k*128
    return kwords
def topk_nearge(wordin,wordout,k=16):
    kin=topk_nearfeats(wordin,k)
    kout=topk_nearfeats(wordout,k)
    print('******',kin,kout)
    result=tf.reduce_mean(tf.reduce_mean(tf.square(kin-kout),axis=-1),axis=[0,1,2])
    return result
    
def normal_process(words,meanword=None,stdword=None):
    meanword1,varword1=tf.nn.moments(words,[1])
    stdword1=tf.sqrt(varword1)
    if meanword!=None and stdword!=None:
        new_words=(words-tf.expand_dims(meanword,axis=1))/(tf.expand_dims(stdword,axis=1)+1e-5)
    else:
        new_words=(words-tf.expand_dims(meanword1,axis=1))/(tf.expand_dims(stdword1,axis=1)+1e-5)
    return new_words,meanword1,stdword1
def cov_loss(meanin,stdin,meanout,stdout):
    result=tf.reduce_mean(tf.abs(stdin/(meanin+1e-5)-stdout/(meanout+1e-5)),[0,1])
    return result
def normalize(words,axis):
    meanword,varword=tf.nn.moments(words,[axis])
    stdword=tf.sqrt(varword)
    new_words=(words-tf.expand_dims(meanword,axis=axis))/(tf.expand_dims(stdword,axis=axis)+1e-5)
    return new_words
#make codeword more sparse
def codelimit_func(codeword):
    #return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(codeword),axis=-1)))
    codeword=tf.cast(codeword,tf.float32)
    return tf.reduce_mean(tf.reduce_sum(codeword,axis=-1))
#batch*2048*3
def get_constrained_part(pointcloud,k,mlp):
    bnum=tf.shape(pointcloud)[0]
    with tf.variable_scope('ad'):
        _,words=adaptive_loss_net(pointcloud,activation_func=tf.nn.relu,n_filter=mlp)
    featnum=words.get_shape()[-1].value
    trans_words=tf.transpose(words,[0,2,1])
    _,ptid=tf.nn.top_k(trans_words,k)#batch*featnum*k
    ptid=tf.reshape(ptid,[-1,featnum*k,1])
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,featnum*k,1])
    idx=tf.concat([bid,ptid],axis=-1)
    selected_pts=tf.gather_nd(pointcloud,idx)
    return selected_pts
#batch*2048,batch*2048
def one_loss(inword,outword,k=2048):
    #inin,_=tf.nn.top_k(tf.squeeze(inword,[2]),k)
    #outout,_=tf.nn.top_k(tf.squeeze(outword,[2]),k)
    inin,_=tf.nn.top_k(inword,k)
    outout,_=tf.nn.top_k(outword,k)
    inin=sumnormalize(inin,'e')
    outout=sumnormalize(outout,'e')
    loss=tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(inin-outout),axis=1)))
    return loss
#batch*2048*16
def separate_loss(words):
    length=words.get_shape()[-1].value
    dismat=tf.square(tf.expand_dims(words,axis=-1)-tf.expand_dims(words,2))+100*tf.reshape(tf.eye(length),[1,1,length,length])#batch*2048*16*16
    #top2dis,_=tf.nn.top_k(-dismat,2)#batch*2048*16*2
    #neardis=tf.reduce_sum(top2dis,axis=-1)
    #print(top2dis,neardis,dismat)
    neardis=tf.reduce_min(dismat,axis=-1)
    result=-tf.reduce_mean(neardis,axis=[0,1,2])
    #meanword,varword=tf.nn.moments(words,[-1])
    #meanval,varval=tf.reduce_mean(meanword,axis=[0,1]),tf.reduce_mean(varword,axis=[0,1])
    return result
def sumnormalize(word,use_type='e'):
    if use_type=='e':
        newword=tf.exp(word)/tf.reduce_sum(tf.exp(word),axis=-1,keepdims=True)
    elif use_type=='n':
        newword=word/tf.reduce_sum(word,axis=-1,keepdims=True)
    return newword
#words:batch*2048*1*128
def distribute_cal(maxword,words,downbound=None,upbound=None,digit=16,precision=None,use_random=False,random_seed=[1.0,1.0]):
    words=tf.expand_dims(words,axis=2)
    minword=tf.reduce_min(words,axis=1,keepdims=True)
    maxword=tf.reshape(maxword,[-1,1,1,maxword.get_shape()[-1].value])
    if precision==None:
        dis=(maxword-minword)*1.0/digit
    else:
        dis=precision
    stones=tf.tile(tf.reshape(tf.linspace(0.0,1.0,digit+1),[1,1,-1,1]),[1,1,1,tf.shape(maxword)[-1]])#batch*1*17*128
    if downbound is None:
        if use_random:
            downbound=tf.tile(minword,[1,1,digit,1])+(maxword-minword)*\
                      (stones[:,:,:-1,:]-tf.random_uniform([BATCH_SIZE,1,digit,tf.shape(maxword)[-1]],0.0,1.0*random_seed[0]/digit))
        else:
            downbound=tf.tile(minword,[1,1,digit,1])+(maxword-minword)*stones[:,:,:-1,:]
    downbound=tf.maximum(downbound,minword)
    if upbound is None:
        if use_random:
            upbound=downbound+tf.tile(dis,[1,1,digit,1])*\
                    tf.random_uniform([BATCH_SIZE,1,digit,tf.shape(maxword)[-1]],0.0,random_seed[1]*dis)#batch*1*16*128
        else:
            print(downbound,dis)
            upbound=downbound+tf.tile(dis,[1,1,digit,1])#batch*1*16*128
    upbound=tf.minimum(upbound,maxword)

    new_words=upbound-(tf.tile(words,[1,1,digit,1])-downbound)
    new_words=tf.nn.relu(new_words)#batch*2048*16*128
    #meanword,varword=tf.nn.moments(words,[1])#batch*1*16*128
    sumword=tf.reduce_sum(new_words,axis=1,keepdims=False)
    maxwords=tf.reduce_max(new_words,axis=1,keepdims=False)
    #varword=tf.sqrt(varword)
    #var_words=tf.reduce_sum()
    
    new_words=tf.cast(tf.cast(new_words,dtype=tf.bool),dtype=tf.float32)
    prob=tf.reduce_sum(new_words,axis=1)/words.get_shape()[1].value
    return prob,downbound,upbound,sumword,maxwords
#batch*1*16*128
def prob_loss(probin,probout,use_type='c'):
    if use_type=='c':
        result=-tf.reduce_mean(tf.reduce_sum(probin*tf.log(probout+1e-5),axis=2),axis=[0,-1])
    elif use_type=='s':
        result=tf.reduce_mean(tf.reduce_sum(tf.square(probin-probout),axis=2),axis=[0,-1])
    return result
#batch*1*1*128
def mean_max(meanfeat,maxfeat,nearzero=False):
    if nearzero:
        result=tf.reduce_mean(meanfeat)
    else:
        result=tf.sqrt(tf.reduce_mean(tf.square(maxfeat-meanfeat)))
    return result
#batch*2048*1*128
def feat_var(words):
    _,varwords=tf.nn.moments(words,[-1])
    result=tf.reduce_mean(tf.sqrt(varwords))
    return result

def max_std(words, use_mean=False):
    if use_mean:
        maxword=tf.reduce_mean(words,axis=1,keepdims=True)
    else:
        maxword=tf.reduce_max(words,axis=1,keepdims=True)
    meanwords=tf.reduce_mean(words,axis=1,keepdims=True)
    subwords=maxword-words
    subwords=tf.reduce_mean(tf.square(subwords),axis=1,keepdims=True)
    max_stdword=tf.sqrt(subwords)
    return meanwords,max_stdword
#batch*16*128
def cross_constraint(upbound,downbound):
    upbound=tf.expand_dims(upbound,axis=2)
    downbound=tf.expand_dims(downbound,axis=2)
    boundnum=upbound.get_shape()[1].value
    featnum=upbound.get_shape()[-1].value
    up2up=upbound-tf.reshape(upbound,[-1,1,boundnum,featnum])
    up2down=upbound-tf.reshape(downbound,[-1,1,boundnum,featnum])
    result=tf.nn.relu(-up2up*up2down)
    return result
#batch*16*1*128
def internal_std(upbound,downbound,words):
    ptnum=words.get_shape()[1].value
    digit=upbound.get_shape()[1].value
    upbound=tf.expand_dims(upbound,axis=1)
    downbound=tf.expand_dims(downbound,axis=1)
    new_words=upbound-(tf.tile(words,[1,1,digit,1])-downbound)
    new_words=tf.nn.relu(new_words)#batch*2048*16*128
    
    words_mask=tf.reduce_min(new_words,axis=-1,keepdims=True)#batch*2048*16*1
    words_mask=words_mask/tf.maximum(words_mask,1e-5)
    region_num=tf.reduce_sum(words_mask,axis=1,keepdims=True)#batch*1*16*1
    region_loss=-region_num/ptnum
     
    cross_num=tf.squeeze(tf.reduce_sum(words_mask,axis=2),[-1])
    cross_num=tf.reduce_sum(cross_num-tf.ones_like(cross_num),axis=1)
    cross_loss=cross_num/ptnum

    filtered_words=new_words*words_mask#batch*2048*16*128
    meanwords=tf.reduce_sum(filtered_words,axis=1,keepdims=True)/region_num#batch*1*16*128
    stdwords=tf.reduce_sum((tf.square(filtered_words-meanwords))*words_mask,axis=1,keepdims=True)/region_num#batch*1*16*128

    return meanwords,stdwords,region_num,cross_num
#batch*16*2*128
def limit2bound(limit,minword,maxword):
    downbound=tf.expand_dims(limit[:,:,0,:],axis=2)#batch*16*1*128
    downbound=tf.maximum(minword,downbound)
    dis=tf.expand_dims(limit[:,:,1,:],axis=2)
    upbound=downbound+dis
    upbound=tf.minimum(maxword,upbound)
    return downbound,upbound
#def sampling(npoint,xyz,use_type='f'):
#    if use_type=='f':
#        idx=tf_sampling.farthest_point_sample(npoint, xyz)
#        new_xyz=tf_sampling.gather_point(xyz,idx)
#    elif use_type=='r':
#        bnum=tf.shape(xyz)[0]
#        ptnum=xyz.get_shape()[1].value
#        ptids=arange(ptnum)
#        random.shuffle(ptids)
#        ptid=tf.tile(tf.constant(ptids[:npoint],shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])
#        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
#        idx=tf.concat([bid,ptid],axis=-1)
#        new_xyz=tf.gather_nd(xyz,idx)
#    return new_xyz
#batch*2048*1*128
def dividing_net(words,mlp=[128,128,2048]):
    ptnum=words.get_shape()[1].value
    limit=local_loss_net(words,activation_func=tf.nn.relu,n_filter=mlp)
    limit=tf.reshape(limit,[-1,ptnum,16,128])
    return limit
def sample_distri(meanval,varval):
    #result=meanval+varval*tf.random_normal(shape=tf.shape(meanval),mean=0.0,stddev=1.0)
    result=meanval+tf.exp(varval/2)*tf.random_normal(shape=tf.shape(meanval),mean=0.0,stddev=1.0)
    return result
def rbf_oper(data,cenpts,ratio=0.1):
    data1=tf.expand_dims(data,axis=2)
    dis=tf.reduce_sum(tf.square(data1-cenpts),axis=-1)#batch*2048*n
    dis=tf.exp(-ratio*dis)
    #dis=tf.concat([data,dis],axis=-1)
    #mask=tf.exp(-decay_factor*dis)
    #newdata=data*tf.expand_dims(mask,axis=-1)#batch*2048*n*3
    return dis
def KLfunc(inmeanval,instdval):
    #result=tf.square(inmeanval)+tf.square(instdval)-2*tf.log(instdval+1e-5)-1
    result=0.1*tf.square(inmeanval)+tf.exp(instdval)-instdval-1
    return result/2
def copy_op(q_scope,target_q_scope):
    t_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
    q_col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
    q_dict = {}
    update_ops = []
    for var in q_col:
        name_index = var.name.find(q_scope)
        var_name = var.name[name_index+len(q_scope):]
        q_dict[var_name] = var
    for var in t_col:
        name_index = var.name.find(target_q_scope)
        var_name = var.name[name_index + len(target_q_scope):]
        update_ops.append(tf.assign(var, q_dict[var_name]))
    final_op = tf.group(*update_ops)
    return final_op
#inpts:b*n*3
def ptspartition(inpts,tnum=4):
    ptnum=np.shape(inpts)[1]
    pmat=inpts[:,:,1]
    #cens=[]
    #for i in range(len(inpts)):
    #    cen=KMeans(n_clusters=tnum, random_state=1, n_init=10,max_iter=300, init='k-means++',tol=1e-4).fit(inpts[i])
    #    cen=cen.cluster_centers_
    #    #cen=np.array(downsam(inpts[i],dtype='u',num=ptnum//tnum,gridsize=1))#4*3
    #   # print(cen[:,0])
    #    #cen=cen[np.argsort(0.01*cen[:,0]+0.1*cen[:,1]+1*cen[:,2],axis=0)]
    #    cens.append(np.expand_dims(cen,axis=0))
    #cens=np.concatenate(cens,axis=0)
    ##print(np.shape(cens))
    #dismat=np.sum(np.square(np.expand_dims(inpts,axis=2)-np.expand_dims(cens,axis=1)),axis=-1)
    ##print(np.shape(dismat))
    #pmat=np.argmin(dismat,axis=-1)
    #print(np.shape(dismat))
    pidx=np.argsort(pmat,axis=-1)
    result=[]
    for i in range(len(inpts)):
        #print(inpts[i,pidx[i]])
        #assert False
        result.append(np.expand_dims(inpts[i,pidx[i]],axis=0))
    result=np.concatenate(result,axis=0)
    print(np.shape(result))
    #assert False
    return result
#pts:b*2048*3
def tfsort(pts,tnum=4):
    ptnum=pts.get_shape()[1].value
    inpts=tf.reshape(pts,[-1,tnum,ptnum//tnum,3])
    cens=tf.reduce_mean(inpts,axis=2)#b*4*3
    #idx,_=get_topk(cens,cens,tnum)#b*4
    _,idx=tf.nn.top_k(-cens[:,:,1],tnum)
    #print (idx)
    #assert False

    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(pts)[0],dtype=tf.int32),[-1,1,1]),[1,tnum,1])
    idx=tf.concat([bid,tf.expand_dims(idx,axis=-1)],axis=-1)
    result=tf.gather_nd(inpts,idx)#batch*n*k*c
    result=tf.reshape(result,[-1,ptnum,3])

    return result
#data:b*n*3
def get_normal(data,cir=True):
    if not cir:
        result=data
        dmax=np.max(result,axis=1,keepdims=True)
        dmin=np.min(result,axis=1,keepdims=True)
        length=np.max((dmax-dmin)/2,axis=-1,keepdims=True)
        center=(dmax+dmin)/2
        result=(result-center)/length
    else:
        cen=np.mean(data,axis=1,keepdims=True)
        rdismat=np.sqrt(np.sum(np.square(data-cen),axis=-1))#b*n
        r=np.max(rdismat,axis=-1,keepdims=True)
        para=1/r
        #print(np.shape(para))
        result=np.expand_dims(para,axis=-1)*(data-cen)#+cen
    return result
def train():
    n_pc_points=PT_NUM
    ptnum=n_pc_points
    bneck_size=128
    featlen=64
    mlp=[64]
    mlp.append(2*featlen)
    mlp2=[128,128]
    cen_num=16
    region_num=1
    gregion=1
    rnum=1
    dim=128
    pointcloud_pl=tf.placeholder(tf.float32,[BATCH_SIZE,PT_NUM,3],name='pointcloud_pl')
    posi_pl=tf.placeholder(tf.float32,[BATCH_SIZE,None,3],name='posi_pl')
    mat_pl=tf.placeholder(tf.float32,[BATCH_SIZE,3,dim],name='mat_pl')

    inpts=pointcloud_pl
    #inpts=get_topk(sampling(4, pointcloud_pl,'f')[-1],pointcloud_pl,PT_NUM//4)[-1]
    #inpts=tf.reshape(inpts,[-1,PT_NUM,3])
    #inpts=tfsort(inpts,4)
    global_step=tf.Variable(0,trainable=False)
    #alpha = tf.train.piecewise_constant(global_step, [1000, 3000, 10000],
    #                                    [0.0001, 0.0001, 0.001, 0.01], 'alpha_op')
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size,mode='ae')
    with tf.variable_scope('ge'):
        #word=dgcnn_kernel(pointcloud_pl, is_training=tf.constant(True), bn_decay=None)
        #_,word=local_kernel(pointcloud_pl,cenlist=None,pooling='max')
        word=encoder(inpts,n_filters=enc_args['n_filters'],filter_sizes=enc_args['filter_sizes'],strides=enc_args['strides'],b_norm=enc_args['b_norm'],verbose=enc_args['verbose'])
        out=decoder(word,layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose'])
    #out=tf.reshape(out,[-1,45*45,3])
    out=tf.reshape(out,[-1,ptnum,3])
    #out=tfsort(out,4)
    #encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points//32, bneck_size,3)
    #with tf.variable_scope('ge'):
    #    cens,feats=local_kernel(pointcloud_pl,pooling='max')
    #    #print(feats)
    #    #assert False
    #    cennum=cens.get_shape()[1].value
    #    outlist=[]
    #    for i in range(cennum):
    #        with tf.variable_scope('dec'+str(i)):
    #            outi=tf.expand_dims(cens[:,i,:],axis=1)\
    #                    +tf.reshape(decoder(feats[:,i,:],layer_sizes=dec_args['layer_sizes'],b_norm=dec_args['b_norm'],b_norm_finish=dec_args['b_norm_finish'],verbose=dec_args['verbose']),[-1,n_pc_points//cennum,3])
    #            outlist.append(outi)
    #    out=tf.concat(outlist,axis=1)
    #with tf.variable_scope('1ad'):
    #    movepts,movelen=movenet(pointcloud_pl,knum=256,mlp1=[128,128],mlp2=[128,128])
    #posis=posi_pl

    #with tf.variable_scope('1ad'):
    #    infeat=find_diff(pointcloud_pl,mlp=[64,128])
    #with tf.variable_scope('1ad',reuse=True):
    #    outfeat=find_diff(out,mlp=[64,128])
    #with tf.variable_scope('1ad',reuse=True):
    #    posifeat=find_diff(posi_pl,mlp=[64,128])
    #
    #ourfeat=tf.concat([infeat,outfeat],axis=-1)
    ##words=ourfeat
    #mlp=[256,256]
    #dim=128
    #with tf.variable_scope('2ad'):
    #    words=fc_layers(ourfeat,dim,mlp)
    #    #for i,outchannel in enumerate(mlp):
    #    #    words=conv2d('basic_state%d'%i,words,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
    #    #words=conv2d('basic_stateoutg',words,dim*3,[1,1],padding='VALID',activation_func=None)
    #matwords=tf.reshape(words[:,:3*dim],[-1,3,dim])
    #biaswords=tf.reshape(words[:,3*dim:],[-1,1,dim])
    #matwords2=matwords#tf.reshape(words,[-1,3,1,dim])
    ##matwords=tf.reshape(words,[-1,1,dim,3])
    ##fi=tf.reduce_mean(tf.nn.leaky_relu(tf.reduce_sum(tf.reshape(pointcloud_pl,[BATCH_SIZE,-1,1,3])\
    ##        *tf.tile(matwords,[1,tf.shape(pointcloud_pl)[1],1,1]),axis=-1)),axis=1)

    ##fr=tf.reduce_mean(tf.nn.leaky_relu(tf.reduce_sum(tf.reshape(out,[BATCH_SIZE,-1,1,3])\
    ##        *tf.tile(matwords,[1,tf.shape(out)[1],1,1]),axis=-1)),axis=1)

    ##fp=tf.reduce_mean(tf.nn.leaky_relu(tf.reduce_sum(tf.reshape(posi_pl,[BATCH_SIZE,-1,1,3])\
    ##        *tf.tile(matwords,[1,tf.shape(posi_pl)[1],1,1]),axis=-1)),axis=1)

    ##fi=tf.reduce_mean(tf.nn.leaky_relu(tf.reduce_sum(tf.reshape(pointcloud_pl,[BATCH_SIZE,3,-1,1])*matwords,axis=1)),axis=1)
    ##fr=tf.reduce_mean(tf.nn.leaky_relu(tf.reduce_sum(tf.reshape(out,[BATCH_SIZE,3,-1,1])*matwords,axis=1)),axis=1)
    ##fp=tf.reduce_mean(tf.nn.leaky_relu(tf.reduce_sum(tf.reshape(posi_pl,[BATCH_SIZE,3,-1,1])*matwords,axis=1)),axis=1)

    ##fii=tf.reduce_mean(tf.reduce_sum(tf.reshape(pointcloud_pl,[BATCH_SIZE,3,-1,1])*mat_pl,axis=1),axis=1)
    ##frr=tf.reduce_mean(tf.reduce_sum(tf.reshape(out,[BATCH_SIZE,3,-1,1])*mat_pl,axis=1),axis=1)
    ##print(fi)
    ##assert False
    ##print(tf.matmul(pointcloud_pl,matwords))
    ##assert False
    ##print(tf.matmul(pointcloud_pl,matwords)+biaswords)

    #fi=tf.reduce_mean(tf.nn.relu(tf.matmul(pointcloud_pl,matwords)+biaswords),axis=1)
    #fr=tf.reduce_mean(tf.nn.relu(tf.matmul(out,matwords)+biaswords),axis=1)
    #fp=tf.reduce_mean(tf.nn.relu(tf.matmul(posi_pl,matwords)+biaswords),axis=1)

    #mfi=tf.reduce_max(tf.nn.relu(tf.matmul(pointcloud_pl,matwords)+biaswords),axis=1)
    #mfr=tf.reduce_max(tf.nn.relu(tf.matmul(out,matwords)+biaswords),axis=1)
    #loss_mri=tf.reduce_mean(tf.reduce_sum(tf.square(mfi-mfr),axis=-1))

    #fii=tf.reduce_sum(tf.nn.leaky_relu(tf.matmul(pointcloud_pl,mat_pl)),axis=1)
    #frr=tf.reduce_sum(tf.nn.leaky_relu(tf.matmul(out,mat_pl)),axis=1)
 


    #fio=tf.concat([infeat,outfeat],axis=-1)
    #fpo=tf.concat([posifeat,outfeat],axis=-1)
    #fip=tf.concat([infeat,posifeat],axis=-1)
    #fiop=tf.concat([infeat,posifeat,outfeat],axis=-1)
    #with tf.variable_scope('2ad'):
    #    inlocal,rinlocal,maxinlocal,lr=local_loss_net(pointcloud_pl,gdifval)
    #with tf.variable_scope('2ad',reuse=True):
    #    outlocal,routlocal,maxoutlocal,_=local_loss_net(out,gdifval,rawmask=maxinlocal,activation_func=None)

    #with tf.variable_scope('1ad'):
    #    movepts,movelen=movenet(pointcloud_pl,posi_pl,knum=128,mlp1=[128,128],mlp2=[128,128])
    with tf.variable_scope('2ad'):
        fi,mfi=global_kernel(pointcloud_pl,mlps=[64,128],acti_func=tf.nn.leaky_relu,pooling='maxmean')
        #fi=clossnet(pointcloud_pl,mlp=[64,64],outlen=128)
    with tf.variable_scope('2ad',reuse=True):
        fr,mfr=global_kernel(out,mlps=[64,128],acti_func=tf.nn.leaky_relu,pooling='maxmean',secword=mfi)
        #fr=clossnet(out,mlp=[64,64],outlen=128)
    with tf.variable_scope('2ad',reuse=True):
        fp,mfp=global_kernel(posi_pl,mlps=[64,128],acti_func=tf.nn.leaky_relu,pooling='maxmean',secword=mfi)
        #fp=clossnet(posi_pl,mlp=[64,64],outlen=128)

    loss_ri=tf.reduce_mean(tf.reduce_sum(tf.abs(fi-fr),axis=-1))
    #loss_pi=tf.reduce_mean(tf.reduce_sum(tf.abs(fp-fi),axis=-1))
    #loss_pi=tf.reduce_mean((tf.reduce_sum(tf.abs(fp-fi),axis=-1,keepdims=True)/tf.reduce_mean(tf.abs(pointcloud_pl-posi_pl),axis=-1)-1)**2)
    loss_pi=tf.reduce_mean(tf.reduce_sum(tf.abs(fp-fi),axis=-1,keepdims=True)/tf.reduce_mean(tf.abs(inpts-posi_pl),axis=-1))  
    loss_pr=tf.reduce_mean(tf.reduce_sum(tf.abs(fp-fr),axis=-1))

    loss_luc=mfi[0]#(mfi[0]+mfr[0]+mfp[0])/3
 
    negdist=tf.reduce_mean(tf.square(tf.tile(tf.expand_dims(fi,axis=1),[1,BATCH_SIZE,1])-tf.expand_dims(fi,axis=0)),axis=-1)#16*16
    filt=tf.ones_like(negdist)-tf.eye(BATCH_SIZE)
    #negdist=-tf.reduce_mean(tf.log(negdist*filt+1e-5))
    negdist=tf.reduce_mean(tf.exp(-negdist*filt))

    #loss_d=tf.exp(-loss_ri)+(loss_pi+loss_pr)+negdist-tf.exp(-loss_mri)#+0.1*loss_mpi
    #loss_d=negdist+0.003*loss_pi
    loss_d=-tf.log(loss_ri+1e-8)+0.003*loss_pi+2*loss_luc#(loss_pi+loss_pr)#-0.1*tf.log(loss_mri+1e-5)+0.1*loss_mpi
            #+(loss_pi+loss_pr)+0.1*negdist#+(loss_mpi+loss_mpr)#+loss_mpi#(loss_pr+loss_pi)+loss_mpi
    #loss_d=0.1*tf.exp(-loss_ri)+0.1*movelen+(loss_pr+loss_pi)
    loss_e=loss_ri#+0.1*loss_pi+0.01*tf.reduce_mean(movelen)#0.01*tf.reduce_mean(routlocal+lr)
    #loss_e=emd_func(pointcloud_pl,out)+multi_chamfer_func(128,pointcloud_pl,out,[4,8,16,32,64])
    #loss_e=chamfer_dcd(pointcloud_pl,out)

    #lamda=0.5
    #lrd=0.00001
    #d_real=pugan_dis('1ad',pointcloud_pl,reuse = tf.AUTO_REUSE,bn = False,start_number = 32)
    #d_fake=pugan_dis('1ad',out,reuse = tf.AUTO_REUSE,bn = False,start_number = 32)

    #lamda=0.05
    #lrd=0.0001
    #d_real=pf_dis('1ad',pointcloud_pl)
    #d_fake=pf_dis('1ad',out)
    
    #lamda=0.005
    #lrd=0.00005
    #d_real=crn_dis('1ad',pointcloud_pl,2)
    #d_fake=crn_dis('1ad',out,2)

    #loss_d=d_loss(d_real,d_fake)
    #local_loss1=g_loss(d_fake)
    #loss_e=chamfer_big(pointcloud_pl,out)+lamda*local_loss1

    trainvars=tf.GraphKeys.TRAINABLE_VARIABLES
    allvars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    varg=tf.get_collection(trainvars,scope='ge')
    varf=tf.get_collection(trainvars,scope='2ad')
    varp=tf.get_collection(trainvars,scope='1ad')
    d_clip= [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in varp]

    bnvar=[v for v in allvars if 'bnorm' in v.name]

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    gezhengze=tf.reduce_sum([regularizer(v) for v in varg])
    pzhengze=tf.reduce_sum([regularizer(v) for v in varp])
    lde_zhengze=tf.reduce_sum([regularizer(v) for v in varf])
    loss_e=loss_e+0.001*gezhengze#+0.0001*pzhengze#//////////////////
    loss_d_local=loss_d+0.0001*lde_zhengze#+0.01*tf.reduce_mean(tf.square(words))
    #d_clip=[v.assign(tf.clip_by_value(v,-0.5,0.5)) for v in var3+var5]
    alldatanum=2048*FILE_NUM
    trainstep=[]
    #print(varf+varp)
    #assert False
    #lr=tf.train.exponential_decay(0.01, global_step, (EPOCH_ITER_TIME*alldatanum)/(BATCH_SIZE*10), 0.96, staircase=True)
    trainstep.append(tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_e, global_step=global_step,var_list=varg+bnvar))
    trainstep.append(tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss_d_local, global_step=global_step,var_list=varf))####################
    #trainstep.append(tf.train.AdamOptimizer(learning_rate=0.01).minimize(1/(local_loss1+0.1)+loss_cons, global_step=global_step,var_list=var3))
    loss=[loss_e,loss_d]

    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.075#0.052#0.06#0.075#0.0765
    config.gpu_options.allow_growth=True

    #train_gen,train_num=get_gen('../dense_data/train.lmdb')
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=20)
        sess.run(tf.global_variables_initializer())
        ivar=[v for v in allvars if v.name.split(':')[0]=='is_training']
        sess.run(tf.assign(ivar[0],True))
        #tf.train.Saver(var_list=varf).restore(sess, tf.train.latest_checkpoint('/apdcephfs/private_yaoshihuang/ours_pnae/'))
        #if os.path.exists('./modelvv_ae/checkpoint'):
        print('here load')
        #saver = tf.train.Saver(var_list=var3+var5).restore(sess, tf.train.latest_checkpoint('./LossNet/PN/fc_lnfc/'))
            #saver.restore(sess, tf.train.latest_checkpoint('./modelvv_ae/'))

        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter("logs/", sess.graph)
        glist=[]
        glist2=[]
        glist3=[]
        #klist=[]
        sig=0
        sig2=0
        lastval=0
        lastlocal=0
        #oastrl=0
        losse,lossd,lossd2=0,0,0
        loss_global,loss_local,localzhengze,meanloss,reverloss,loss_local1,loss_local2=0,0,0,0,0,0,0
        cyclenum=1
        rlin=100
        reverloss=100
        lastrl=0
        import time
        #tlist=[]
        datalist=[]
        #wmat=[]
        import copy
        for j in range(FILE_NUM):
            traindata = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))[:,:,:3]
            traindata = get_normal(traindata,True)
            #traindata = ptspartition(traindata)
            datalist.append(traindata)
            #wmat.append(np.ones(len(traindata)))
            #print(j)
        #print(len(datalist))
        path='./pn_newae.txt'
        if os.path.exists(path):
            os.remove(path)
        tlist=[]
        for i in range(EPOCH_ITER_TIME):
            #if (i+1)%5==0:
            #    for j in range(FILE_NUM):
            #        wmat[j]=np.ones_like(wmat[j])
            for j in range(FILE_NUM):
                traindata=copy.deepcopy(datalist[j])
                #traindata=datalist[0]
                #traindata0 = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))
                #print(np.mean(np.abs(traindata-traindata0)))
                #ptspartition(traindata)
                #print(np.max(traindata,axis=1),np.min(traindata,axis=1))
                #assert False
                #_,_,_,traindata=next(train_gen)
                
                ids=list(range(len(traindata)))
                random.shuffle(ids)
                traindata=traindata[ids,:,:]

                #wmat[j]=wmat[j][ids]
                
                allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
                batch_num=int(allnum/BATCH_SIZE)
                glist=[]
                dlist=[]
                #batch_num=train_num//BATCH_SIZE 
                
                #labs=np.ones(batch_num)
                #labs[:int(0.4*batch_num)]=0
                #random.shuffle(labs)
                
                #prob=np.sum(np.reshape(wmat[j][:(len(wmat[j])//BATCH_SIZE)*BATCH_SIZE],[-1,BATCH_SIZE]),axis=-1)
                #prob=np.minimum(np.maximum((max(prob)-prob)/(max(prob)-min(prob)+1e-5),0.4),1.0)
                #print(prob)

                for batch in range(batch_num):
                    start_idx = (batch * BATCH_SIZE) % allnum
                    end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                    batch_point=copy.deepcopy(traindata[start_idx:end_idx])
                    #batch_point=traindata[:BATCH_SIZE]
                    batch_point=shuffle_points(batch_point)
                    posis=batch_point

                    #posis=batch_point[:,:int(np.clip(np.random.rand(1),1/2,1)*np.shape(batch_point)[1])]
                    posis=jitter_point_cloud(posis,0.003,0.2)
                    #random.shuffle(batch_point)
                    #_,_,_,batch_point=next(train_gen)
                    for ei in range(cyclenum):
                        feed_dict = {pointcloud_pl: batch_point[:,:PT_NUM,:],posi_pl:posis}
                        #if (batch+1) % 1 == 0:
                        #    sess.run(adcopy)
                        resi=[0,[0,0]]
                        stime=time.time()
                        #losse=sess.run(loss_e, feed_dict=feed_dict)
                        #if len(glist)>1 and len(dlist)>1:
                        #    prob=3*(dlist[-1]-glist[-1])/dlist[-1]
                        #    print(prob)
                        #else:
                        #    prob=0.5
                        #wmat[j]=wmat[j][start_idx:end_idx]+np.ones(BATCH_SIZE)
                        val=np.random.rand()
                        #if val<1.0*prob[batch] and val>0.1:
                        ##if labs[batch]==1:
                        ##if len(glist)==0 or glist[-1]<=0.5*dlist[-1]:
                        #sess.run([trainstep[1],d_clip], feed_dict=feed_dict)
                        #    wmat[j][start_idx:end_idx]=wmat[j][start_idx:end_idx]+np.ones(BATCH_SIZE)
                        #elif val<=0.1:
                        sess.run([trainstep[1]], feed_dict=feed_dict)
                        #    resi = sess.run([trainstep[0]], feed_dict=feed_dict)
                        #else:
                        #feed_dict[mat_pl]=mymat
                        resi = sess.run([trainstep[0],[fp,fr,fi],loss_e], feed_dict=feed_dict)
                            #glist.append(resi[1])
                        #resi = sess.run([trainstep[1],trainstep[0],loss], feed_dict=feed_dict)
                        #resi = sess.run([trainstep[0]], feed_dict=feed_dict)
                        etime=time.time()
                        if i>0:
                            tlist.append(etime-stime)
                        #losse,lossd=resi[1]
                        #glist.append(lossd)
                        #glist2.append(lossd)
                        #glist3.append(rlin)
                    if (batch+1) % 16 == 0:
                        #path='/apdcephfs/private_yaoshihuang/rrcae.txt'
                        #if os.path.exists(path):
                        #    os.remove(path)
                        wf=open(path,'a')
                        wf.write('mean time'+str(mean(tlist))+'\n')
                        fpv,frv,fiv=resi[1]
                        wf.write('epoch: %d '%i+'file: %d '%j+'batch: %d' %batch+'\n')
                        wf.write('loss: '+str(resi[-1])+'\n')
                        wf.close()
            #print('mean time', mean(tlist))
                        
            if (i+1)%100==0:
                print('mean time', mean(tlist))
                #assert False
                save_path = saver.save(sess, './ours_newae/model',global_step=i)
if __name__=='__main__':
    train()
