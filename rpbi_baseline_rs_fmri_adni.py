# -*- coding: utf-8 -*-
"""
RPBI of pairwise DX groups.
Plot t-maps and p-maps on the voxels.
@author: Mehdi
"""

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.plotting import plot_roi, plot_stat_map, plot_img
from nilearn.mass_univariate import randomized_parcellation_based_inference
from matplotlib import cm
from fetch_data import datasets

CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi',
                         'data', 'tmp')

dataset = datasets.fetch_adni_rs_fmri()
func_files = dataset['func']
dx_group = np.array(dataset['dx_group'])

###############################################################################
# 1- Masking
###############################################################################
masker = MultiNiftiMasker(mask_strategy='epi',
                          mask_args=dict(opening=1),
                          memory_level=2,
                          memory=CACHE_DIR,
                          n_jobs=8)
func_masked = masker.fit_transform(func_files)

###############################################################################
#2- Testing
###############################################################################
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]

for gr in groups:

    test_var = np.ones((len(func_files), 1), dtype=float)  # intercept

    neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    test_var, func_masked,  # + intercept as a covariate by default
    np.asarray(masker.mask_img_.get_data()).astype(bool),
    n_parcellations=100,  # 30 for the sake of time, 100 is recommended
    n_parcels=500,
    threshold=0,
    n_perm=10000,  # 1,000 for the sake of time. 10,000 is recommended
    memory=CACHE_DIR,
    n_jobs=20, verbose=False)
        
    neg_log_pvals_rpbi_unmasked = masker.inverse_transform(
    np.ravel(neg_log_pvals_rpbi))

    p_path = os.path.join(CACHE_DIR, 'figures',
                          'pmap_rpbi_'+gr[0]+'_'+gr[1]+'_baseline_fmri_adni')
    neg_log_pvals_rpbi_unmasked.to_filename(p_path+'.nii.gz')
    break