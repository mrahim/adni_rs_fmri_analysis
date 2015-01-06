# -*- coding: utf-8 -*-
"""
RPBI of pairwise DX groups.
Plot t-maps and p-maps on the voxels.
@author: Mehdi
"""

# 1- Masking data
# 2- T-Testing

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, plot_stat_map, plot_img
from nilearn.mass_univariate import randomized_parcellation_based_inference
from matplotlib import cm





nb_vox = pet_masked.shape[1]

groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]

for gr in groups:
    gr1_idx = data[data.DX_Group == gr[0]].index.values
    gr2_idx = data[data.DX_Group == gr[1]].index.values
    
    gr1_f = pet_masked[gr1_idx, :]
    gr2_f = pet_masked[gr2_idx, :]
    
    gr_idx = np.hstack([gr1_idx, gr2_idx])
    gr_f = pet_masked[gr_idx, :]
    gr_labels = np.vstack([np.hstack([[1]*len(gr1_idx), [0]*len(gr2_idx)]),
                           np.hstack([[0]*len(gr1_idx), [1]*len(gr2_idx)])]).T

    test_var = np.hstack([[1]*len(gr1_idx), [0]*len(gr2_idx)])
    
    neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    test_var, gr_f,  # + intercept as a covariate by default
    np.asarray(masker.mask_img_.get_data()).astype(bool),
    n_parcellations=200,  # 30 for the sake of time, 100 is recommended
    n_parcels=500,
    threshold=0,
    n_perm=10000,  # 1,000 for the sake of time. 10,000 is recommended
    n_jobs=4, verbose=False)
    
    neg_log_pvals_rpbi_unmasked = masker.inverse_transform(
    np.ravel(neg_log_pvals_rpbi))

    p_path = os.path.join('figures',
                          'pmap_rpbi_voxel_norm_'+gr[0]+'_'+gr[1]+'_baseline_adni')
    plot_stat_map(neg_log_pvals_rpbi_unmasked, img, output_file=p_path,
                  black_bg=True, title='/'.join(gr))    
    neg_log_pvals_rpbi_unmasked.to_filename(p_path+'.nii')