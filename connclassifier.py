# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:48:41 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob
import numpy as np
from fetch_data import fetch_adni_masks, fetch_adni_rs_fmri, \
                       set_cache_base_dir, set_group_indices
from nilearn.input_data import NiftiMapsMasker
from nilearn.datasets import fetch_msdl_atlas
from sklearn.covariance import GraphLassoCV, LedoitWolf, OAS, \
                               ShrunkCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.base import Bunch
from sklearn.cross_validation import StratifiedShuffleSplit

from joblib import Parallel, delayed
CACHE_DIR = set_cache_base_dir()

def compute_connectivity_subject(conn, func, masker):
    """ Returns connectivity of one fMRI for a given atlas
    """
    ts = masker.fit_transform(func)
    
    if conn == 'gl':
        fc = GraphLassoCV()
    elif conn == 'lw':
        fc = LedoitWolf()
    elif conn == 'oas':
        fc = OAS()
    elif conn == 'scov':
        fc = ShrunkCovariance()
        
    fc.fit(ts)
    ind = np.tril_indices(ts.shape[1], k=-1)
    return fc.covariance_[ind], fc.precision_[ind]

def compute_connectivity_subjects(func_list, atlas, mask, conn, n_jobs):
    """ Returns connectivities for all subjects
    tril matrix n_subjects * n_rois_tril
    """
    masker = NiftiMapsMasker(maps_img=atlas, mask_img=mask,
                             detrend=True, low_pass=.1, high_pass=.01, t_r=3.,
                             resampling_target='data',
                             memory=CACHE_DIR, memory_level=2)

    p = Parallel(n_jobs=n_jobs, verbose=5)(delayed(compute_connectivity_subject)\
                                            (conn, func, masker)\
                                            for func in func_list)
    return np.asarray(p)

###############################################################################
# Connectivity
###############################################################################
dataset = fetch_adni_rs_fmri()
mask = fetch_adni_masks()['mask_petmr']
atlas = fetch_msdl_atlas()['maps']
conn_name= 'lw'
conn = compute_connectivity_subjects(list(dataset.func), atlas, mask,
                                     conn=fc, n_jobs=20)

###############################################################################
# Classification
###############################################################################
def train_and_test(classifier, X, y, train, test):
    """ Returns accuracy and coeffs for a train and test
    """
    classifier.fit(X[train], y[train])
    score = classifier.score(X[test], y[test])
    B = Bunch(score=score, coef=classifier.coef_)
    return B


idx = set_group_indices(dataset.dx_group)
idx['MCI'] = np.hstack((idx['EMCI'], idx['LMCI']))

groups = ['AD', 'MCI']
groups_idx = np.hstack((idx[groups[0]], idx[groups[1]]))
X = conn[groups_idx, :]
y = np.asarray([1] * len(idx[groups[0]]) + [0] * len(idx[groups[1]]))

classifier = LogisticRegression(penalty='l1', random_state=42)
sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.25, random_state=42)

p = Parallel(n_jobs=20, verbose=5)(delayed(train_and_test)(classifier, X, y,
             train, test) for train, test in sss)

output_file = os.path.join(CACHE_DIR, 'lw')
np.savez_compressed(data=p)