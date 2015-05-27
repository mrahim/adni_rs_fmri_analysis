# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:40:11 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_masks, fetch_adni_rs_fmri, \
                       set_cache_base_dir, set_group_indices, \
                       fetch_adni_baseline_rs_fmri
from nilearn.datasets import fetch_msdl_atlas
from nilearn.plotting import plot_connectome
from scipy.stats import ttest_1samp

CACHE_DIR = set_cache_base_dir()

###############################################################################
# Atlas
###############################################################################

def fetch_atlas(atlas_name):
    """Retruns selected atlas path
    """
    if atlas_name == 'msdl':
        atlas = fetch_msdl_atlas()['maps']
    elif atlas_name == 'harvard_oxford':
#        atlas = os.path.join(CACHE_DIR, 'atlas',
#                             'HarvardOxford-cortl-prob-2mm.nii.gz')
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'HarvardOxford-cortl-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'juelich':
#        atlas = os.path.join(CACHE_DIR, 'atlas',
#                             'Juelich-prob-2mm.nii.gz')
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'Juelich-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'mayo':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_68_rois.nii.gz')
    elif atlas_name == 'canica':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_canica_61_rois.nii.gz')
    return atlas


def atlas_to_coords(atlas_name):
    """Returns coords of atlas ROIs
    """

    atlas = fetch_atlas('harvard_oxford')
    affine = nib.load(atlas).get_affine()
    data = nib.load(atlas).get_data()

    centroids = []
    if len(data.shape) == 4:
        for i in range(data.shape[-1]):
            centroid = np.mean(np.where(data[..., i] == 1), axis=1)
            centroid = np.append(centroid, 1)
            centroid = np.dot(affine, centroid)[:-1]
            centroids.append(centroid)
    else:
        vals = np.unique(data)
        for i in range(len(vals)):
            centroid = np.mean(np.where(data == i), axis=1)
            centroid = np.append(centroid, 1)
            centroid = np.dot(affine, centroid)[:-1]
            centroids.append(centroid)

    centroids = np.asarray(centroids)
    return centroids


###############################################################################
# coeffs
###############################################################################

def retrieve_coeffs(atlas_name, metric, classifier_name, group, n_rois):
    ''' boxplot accuracies
    '''
    
    fname = '_'.join([group[0], group[1], atlas_name, metric, classifier_name])
    input_file = os.path.join(BASE_DIR, fname + '.npz')

    if not os.path.isfile(input_file):
        print 'not found'
        return 0
    else:
        data = np.load(input_file)['data']
        coeffs = []
        for d in data:
            coeffs.append(d.coef[0, :])
    coeffs = np.asarray(coeffs)
    
    # T-TEST
    threshold = .05
    threshold /= n_rois*(n_rois-1)/2.    
    thresh_log = -np.log10(threshold)
    tv, pv = ttest_1samp(coeffs, 0.)
    
    # Convert in log-scale
    pv = -np.log10(pv)
    #Locate unsignificant tests
    ind_threshold = np.where(pv < thresh_log)
            
    coeffs = np.mean(coeffs, axis=0)    
    #and then threshold
    coeffs[ind_threshold] = 0    
    
    ind = np.tril_indices(n_rois, k=-1)
    m_coeffs = np.zeros((n_rois, n_rois))
    m_coeffs[ind] = coeffs
    
    m_coeffs = (m_coeffs + m_coeffs.T) / 2.
    
    return m_coeffs


atlas_names = ['canica', 'mayo', 'juelich', 'harvard_oxford']
atlas_name = 'harvard_oxford'
metric = 'oas'
classifier_name = 'ridge'
groups = [['AD', 'MCI'], ['AD', 'Normal'], ['MCI', 'Normal']]

BASE_DIR = '/disk4t/mehdi/data/tmp/conn_' + atlas_name
centroids = atlas_to_coords(atlas_name)
adj_mat = retrieve_coeffs(atlas_name, metric, classifier_name, groups[0], len(centroids))
plot_connectome(adj_mat , centroids, edge_threshold='99.8%')
