# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 13:18:47 2015

@author: Mehdi Rahim
"""
import os
from fetch_data import datasets
import numpy as np
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from nilearn.input_data import NiftiMasker

CACHE_DIR = '/disk4t/mehdi/data/tmp'

def ward_adni_rs_fmri(func_files, n_clusters=200):

    masker = NiftiMasker(mask_strategy='epi',
                         mask_args=dict(opening=1))
    masker.fit(func_files)
    func_masked = masker.transform(func_files)
    #func_masked = masker.transform_niimgs(func_files, n_jobs=4)
    func_masked = np.vstack(func_masked)
    
    ###########################################################################
    # Ward
    ###########################################################################
    
    mask = masker.mask_img_.get_data().astype(np.bool)
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                       n_z=shape[2], mask=mask)
    
    # Computing the ward for the first time, this is long...
    ward = WardAgglomeration(n_clusters=n_clusters, connectivity=connectivity,
                             memory='nilearn_cache')
    
    ward.fit(func_masked)
    ward_labels_unique = np.unique(ward.labels_)
    ward_labels = ward.labels_
        
    ward_filename = '_'.join(['ward', str(n_clusters)])
    img_ward = masker.inverse_transform(ward.labels_)
    img_ward.to_filename(os.path.join(CACHE_DIR, ward_filename))
    
    
    
dataset = datasets.fetch_adni_rs_fmri()
func_files = dataset['func']
n_sample = 140
idx = np.random.randint(len(func_files), size=n_sample)
func_files_sample = np.array(func_files)[idx][0]

ward_adni_rs_fmri(func_files_sample)