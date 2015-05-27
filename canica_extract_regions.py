# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:39:53 2015

@author: mehdi.rahim@cea.fr
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_roi
from nilearn.image import index_img
from fetch_data import fetch_adni_masks, array_to_niis, array_to_nii
from scipy.ndimage import label
from matplotlib import cm
import nibabel as nib
from fetch_data import set_cache_base_dir
from joblib import Parallel, delayed

base_dir = os.path.join(set_cache_base_dir(), 'decomposition')
mask = fetch_adni_masks()['mask_petmr']
mask_shape = nib.load(mask).shape
mask_affine = nib.load(mask).get_affine()
np_files = os.listdir(base_dir)


def extract_region_i(maps, i):
    """ Extract ROIs and plot
    """
    m = maps[i, ...]
    th_value = np.percentile(m, 100.-(100./42.))
    data = np.absolute(array_to_nii(m, mask).get_data())
    data[data <= th_value] = 0
    data[data > th_value] = 1
    data_lab = label(data)[0]
    
    for v in np.unique(data_lab):
        if len(np.where(data_lab == v)[0]) < 1000:
            data_lab[data_lab == v] = 0
        
    img_l = nib.Nifti1Image(data_lab, mask_affine)
    plot_roi(img_l, title=map_title + '_roi_' + str(i), cmap=cm.rainbow)
    plot_stat_map(index_img(img, i), title=map_title + '_' + str(i),
                  threshold=0)


output_label_list = []

for np_file in np_files:
    data = np.load(os.path.join(base_dir, np_file))
    map_title = 'canica'
    if os.path.splitext(np_file)[0] != 'canica' and os.path.splitext(np_file)[0] != 'canica_200':
        map_title = 'tv_msdl'
        data = data['subject_maps']
    print data.shape
    maps = data
    img = array_to_niis(maps, mask)

#    Parallel(n_jobs=21, verbose=5)(delayed(extract_region_i)(maps, i)
#                                    for i in range(maps.shape[0]))
    
    for i in range(maps.shape[0]):
        m = maps[i, ...]
        th_value = np.percentile(m, 100.-(100./42.))
        data = np.absolute(array_to_nii(m, mask).get_data())
        data[data <= th_value] = 0
        data[data > th_value] = 1
        data_lab = label(data)[0]
        
        for v in np.unique(data_lab):
            if len(np.where(data_lab == v)[0]) < 250:
                data_lab[data_lab == v] = 0
        
        for v in np.unique(data_lab):
            data_labels = np.copy(data_lab)
            if v > 0:
                data_labels[data_labels != v] = 0
                data_labels[data_labels == v] = 1
                output_label_list.append(data_labels)
            
        img_l = nib.Nifti1Image(data_lab, mask_affine)
        plot_roi(img_l, title=map_title + '_roi_' + str(i), cmap=cm.rainbow, output_file='canica130/'+map_title + '_roi_' + str(i))
        plot_stat_map(index_img(img, i), title=map_title + '_' + str(i), output_file='canica130/'+map_title + '_stat_' + str(i),
                      threshold=0)
    
    break