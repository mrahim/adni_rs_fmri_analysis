# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:34:59 2015

@author: mehdi.rahim@cea.fr
"""


from base_network_connectivity import NetworkConnectivity, atlas_rois_to_coords
import os
import numpy as np
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, set_cache_base_dir
from nilearn.plotting import plot_connectome

CACHE_DIR = set_cache_base_dir()
dataset = fetch_adni_baseline_rs_fmri()
mask = fetch_adni_masks()['mask_fmri']

atlas_name = 'msdl'
metric = 'corr'

rois = np.arange(3, 7)
rois_names = ['L-DMN', 'M-DMN', 'F-DMN', 'R-DMN']
n_rois = len(rois)
centroids = atlas_rois_to_coords(atlas_name, rois)

conn = NetworkConnectivity(atlas_name=atlas_name, rois=rois, metric=metric,
                           mask=mask, memory=CACHE_DIR, n_jobs=20)
fc = conn.fit(dataset.func)

ind = np.tril_indices(n_rois, k=-1)
m = np.zeros((n_rois, n_rois))
m[ind] = fc[1, ...]
m = .5 * (m + m.T)
plot_connectome(m, centroids)