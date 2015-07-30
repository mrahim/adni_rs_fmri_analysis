# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:34:59 2015

@author: mehdi.rahim@cea.fr
"""


from base_connectivity import Connectivity
import numpy as np
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, \
                        set_cache_base_dir, set_group_indices, fetch_adni_longitudinal_rs_fmri_DARTEL
from nilearn.plotting import plot_connectome
from scipy import stats

CACHE_DIR = set_cache_base_dir()
dataset = fetch_adni_baseline_rs_fmri()
#dataset = fetch_adni_longitudinal_rs_fmri_DARTEL()
mask = fetch_adni_masks()['mask_fmri']

atlas_name = 'mayo'
metric = 'correlation'

conn = Connectivity(atlas_name=atlas_name, rois=True, metric=metric, mask=mask, detrend=True,
                    low_pass=.1, high_pass=.01, t_r=3.,
                    resampling_target='data', smoothing_fwhm=6.,
                    memory=CACHE_DIR, memory_level=2, n_jobs=20)

fc = conn.fit(dataset.func)
np.savez_compressed('longitudinal_dartel_fc', data=fc)

idx = set_group_indices(dataset.dx_group)
groups = [['AD', 'MCI'], ['AD', 'Normal'], ['MCI', 'Normal']]


for g in groups:
    t, p = stats.ttest_ind(fc[idx[g[0]]], fc[idx[g[1]]])
    tv = t
    tv[np.where(p>.01)] = 0

    n_rois = conn.rois['n_rois']
    centroids = conn.rois['rois_centroids']
    ind = np.tril_indices(n_rois, k=-1)
    m = np.zeros((n_rois, n_rois))
    m[ind] = tv
    m = .5 * (m + m.T)
    plot_connectome(m, centroids, title='_'.join(g))
    