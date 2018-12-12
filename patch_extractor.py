#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import nrrd
import math
from random import sample

def dataset_loader(file_path,pid,postfix):
#
#   loads dataset from nrrd files in a list of  lists
#
#   file_path: str - nrrd files path
#   pid: list - patient ID
#   postfix: str - file name postfix
    dataset=[]
    for j in range (0,len(pid)):
        data, options = nrrd.read(file_path + pid[j]  + postfix)
        data = np.transpose(data,(2,0,1))# (number of slices, x, y)
        dataset.append(data)
    return dataset

def get_roi_slices_map(data):
#    data: list of lists of roi per patient
    dataset=[] 
    for i in range (0,len(data)):
        patient=[]
        for j in range(0,len(data[i])):
            if data[i][j][~np.all(data[i][j]==0, axis=0)].shape[0]>0: #if roi exists
                patient.append(True) #roi slices per patient
            else:
                patient.append(False) # no-roi slices per patient
        dataset.append(patient) #add patient to dataset
    return dataset # roi maps for every patient

def merge_roi_ds(dataset, roi, roi_map):
#   dataset: list of images per patient
#   roi: list of images roi
#   roi_map: list with booleans of tumor existance
    merged = []
    for i in range (0,len(dataset)):#patient
        for j in range(0,len(dataset[i])):#slice
            if roi_map[i][j] == True:
                image = np.reshape(dataset[i][j],[1,dataset[i][j].shape[0],dataset[i][j].shape[1]])
                new_roi = np.reshape(roi[i][j],[1,roi[i][j].shape[0],roi[i][j].shape[1]])
                new_image = np.concatenate((image,new_roi), axis=0) # merge image with roi as color channel
                merged.append(new_image)
    return merged

def get_patches(images,roi,patch_size=32,thresshold=0.4,oversampling=1, fast=True):
#    dataset_multi : dict with images, roi (3d roi volume), roi_map (which 2d slices per patient include roi)
#    patch_size: int - 32:[32,32]
#    thresshold: how much ROI in patch
#   oversampling: % more samples from no-roi patches
#   fast: if True patch extraction only from slices with roi
    patches_ace=[] #list of patches with roi
    patches_zero=[] #list of patches without roi
    roi_map = get_roi_slices_map(roi)
    if fast == False:
        for i in range(0, len(roi_map)):
            for j in range(0, len(roi_map[i])):
                roi_map[i][j] = True
    merged = merge_roi_ds(images,roi,roi_map)
    merged = np.stack(merged)# 4D tensor [patches, ch, row, col]
    merged = np.transpose(merged,(0,2,3,1))# 4D tensor [patches, row, col, ch]
	
	#patch extraction
    with tf.Session() as sess:
        with sess.as_default():
            batchTF = patch_extractor(merged,patch_size=patch_size,channels=2)
            batch = batchTF.eval(session=sess)
    for i in range (0,batch.shape[0]):
        if np.count_nonzero(batch[i][1]) > math.floor(patch_size*patch_size*thresshold):
            patches_ace.append(batch[i][0])
        elif np.count_nonzero(batch[i][1]) == 0:
            patches_zero.append(batch[i][0])
    zero_max_elements = int(oversampling*len(patches_ace)) #patches with no-roi will be more
    patches_zero = sample(patches_zero, zero_max_elements) #random sample selection
    return patches_ace,patches_zero

def patch_extractor(images,patch_size=32,channels=2, padding ="SAME"):
#       images: [batch, rows, columns, depth]
#       ksizes: [1,rows,columns,depth]
#       strides: the distance between the two consecutive patches
#       rates: [1,1,1,1] - how far two consecutive patch samples
#       padding: "VALID" or "SAME"
    stride = patch_size//2
    ksizes = [1, patch_size, patch_size, 1]
    strides = [1, stride, stride, 1]
    rates = [1,1,1,1]


    with tf.name_scope('patch_extraction'):#tf.name_scope creates namespace for operators in the default graph.
        patches = tf.extract_image_patches(
                images=images,# 4D tensor
                ksizes=ksizes,
                strides=strides,
                rates=rates,
                padding=padding)
        patches_shape = tf.shape(patches)
        return tf.transpose(tf.reshape(patches, [tf.reduce_prod(patches_shape[0:3]), patch_size, patch_size, int(channels)]),perm=[0,3,1,2])