# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:34:59 2015

@author: mehdi.rahim@cea.fr
"""


from base_network_connectivity import NetworkConnectivity, atlas_rois_to_coords
from base_connectivity import Connectivity
import os
import numpy as np
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, set_cache_base_dir
from nilearn.plotting import plot_connectome

CACHE_DIR = set_cache_base_dir()
dataset = fetch_adni_baseline_rs_fmri()
mask = fetch_adni_masks()['mask_fmri']

atlas_name = 'mayo'
metric = 'corr'




conn = Connectivity(atlas_name=atlas_name, rois=True, metric=metric, mask=mask, detrend=True,
                    low_pass=.1, high_pass=.01, t_r=3.,
                    resampling_target='data', smoothing_fwhm=6.,
                    memory=CACHE_DIR, memory_level=2, n_jobs=20)

fc = conn.fit(dataset.func[:1])

#ind = np.tril_indices(n_rois, k=-1)
#m = np.zeros((n_rois, n_rois))
#m[ind] = fc[1, ...]
#m = .5 * (m + m.T)
#plot_connectome(m, centroids)