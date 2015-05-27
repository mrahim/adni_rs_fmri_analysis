# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:48:41 2015

@author: mehdi.rahim@cea.fr
"""

import os, sys
import numpy as np
import nibabel as nib
from fetch_data import fetch_adni_masks, fetch_adni_rs_fmri, \
                       set_cache_base_dir, set_group_indices, \
                       fetch_adni_baseline_rs_fmri
from nilearn.input_data import NiftiMapsMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_msdl_atlas
from sklearn.covariance import GraphLassoCV, LedoitWolf, OAS, \
                               ShrunkCovariance
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.datasets.base import Bunch
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from joblib import Parallel, delayed
CACHE_DIR = set_cache_base_dir()

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

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
    elif atlas_name == 'canica141':
	atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_canica_141_rois.nii.gz')
    elif atlas_name == 'tvmsdl':
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_tv_msdl.nii.gz')
    return atlas


###############################################################################
# Connectivity
###############################################################################
from scipy import stats, linalg
 
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.


    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
 
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr



def compute_connectivity_subject(conn, func, masker):
    """ Returns connectivity of one fMRI for a given atlas
    """
    ts = masker.fit_transform(func)
    
    if conn == 'gl':
        fc = GraphLassoCV(max_iter=1000)
    elif conn == 'lw':
        fc = LedoitWolf()
    elif conn == 'oas':
        fc = OAS()
    elif conn == 'scov':
        fc = ShrunkCovariance()
    elif conn == 'corr' or conn == 'pcorr':
	fc = Bunch(covariance_=0, precision_=0)
    
    if conn == 'corr' or conn == 'pcorr':
        fc.covariance_ = np.corrcoef(ts)
	fc.precision_ = partial_corr(ts)
    else:
	fc.fit(ts)
    ind = np.tril_indices(ts.shape[1], k=-1)
    return fc.covariance_[ind], fc.precision_[ind]

def compute_connectivity_subjects(func_list, atlas, mask, conn, n_jobs=-1):
    """ Returns connectivities for all subjects
    tril matrix n_subjects * n_rois_tril
    """
    if len(nib.load(atlas).shape) == 4:
        masker = NiftiMapsMasker(maps_img=atlas, mask_img=mask,
                                 detrend=True, low_pass=.1, high_pass=.01, t_r=3.,
                                 resampling_target='data', smoothing_fwhm=6,
                                 memory=CACHE_DIR, memory_level=2)
    else:
        masker = NiftiLabelsMasker(labels_img=atlas, mask_img=mask, t_r=3.,
                                   detrend=True, low_pass=.1, high_pass=.01, 
                                   resampling_target='data', smoothing_fwhm=6,
                                   memory=CACHE_DIR, memory_level=2)

    p = Parallel(n_jobs=n_jobs, verbose=5)(delayed(
                 compute_connectivity_subject)(conn, func, masker)\
                 for func in func_list)
    return np.asarray(p)


###############################################################################
# Classification
###############################################################################
def train_and_test(classifier, X, y, train, test):
    """ Returns accuracy and coeffs for a train and test
    """
    classifier.fit(X[train, :], y[train])
    score = classifier.score(X[test, :], y[test])
    B = Bunch(score=score, coef=classifier.coef_)
    return B

def classify_connectivity(X, y, classifier_name, n_jobs=-1):
    """ Returns 100 shuffle split scores
    """
    if classifier_name == 'logreg_l1':
        classifier = LogisticRegression(penalty='l1', dual=False,
                                        random_state=42)
    elif classifier_name == 'logreg_l2':
        classifier = LogisticRegression(penalty='l2', random_state=42)
    elif classifier_name == 'ridge':
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    elif classifier_name == 'svc_l2':
        classifier = LinearSVC(penalty='l2', random_state=42)
    elif classifier_name == 'svc_l1':
        classifier = LinearSVC(penalty='l1', dual=False, random_state=42)

    p = Parallel(n_jobs=n_jobs, verbose=5)(delayed(train_and_test)(
                 classifier, X, y, train, test) for train, test in sss)
    return p    


###############################################################################
# Main loop
###############################################################################
#dataset = fetch_adni_rs_fmri()
dataset = fetch_adni_baseline_rs_fmri()

mask = fetch_adni_masks()['mask_petmr']

atlas_names = ['canica141', 'canica', 'mayo', 'harvard_oxford', 'juelich', 'msdl']
atlas_names = ['tvmsdl']
for atlas_name in atlas_names:
    atlas = fetch_atlas(atlas_name)
    conn_names = ['gl', 'lw', 'oas', 'scov']
    #conn_names = ['corr']
    for conn_name in conn_names:
        conn = compute_connectivity_subjects(list(dataset.func), atlas, mask,
                                             conn=conn_name, n_jobs=-1)
        idx = set_group_indices(dataset.dx_group)
        idx['MCI'] = np.hstack((idx['EMCI'], idx['LMCI']))
        all_groups = [['AD', 'MCI'], ['AD', 'Normal'], ['MCI', 'Normal']]
    
        for groups in all_groups:
            groups_idx = np.hstack((idx[groups[0]], idx[groups[1]]))
            X = conn[groups_idx, 0, :]
	    #X = StandardScaler().fit_transform(X)
            y = np.asarray([1] * len(idx[groups[0]]) +
                           [0] * len(idx[groups[1]]))
            sss = StratifiedShuffleSplit(y, n_iter=100,
                                         test_size=.25, random_state=42)
        
            classifier_names = ['ridge', 'svc_l1', 'svc_l2',
                                'logreg_l1', 'logreg_l2']
    
            for classifier_name in classifier_names:
                print atlas_name, conn_name, groups, classifier_name    
                p = classify_connectivity(X, y, classifier_name)
                output_folder = os.path.join(CACHE_DIR,
                                             '_'.join(['conn', atlas_name]))
                if not os.path.isdir(output_folder):
                    os.mkdir(output_folder)
                output_file = os.path.join(output_folder,
                                           '_'.join([groups[0], groups[1],
                                                     atlas_name, conn_name,
                                                     classifier_name]))
                np.savez_compressed(output_file, data=p)
