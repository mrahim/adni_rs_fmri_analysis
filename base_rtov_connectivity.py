# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:08:19 2015

@author: mehdi.rahim@cea.fr
"""

###############################################################################
# Connectivity
###############################################################################

import os
import numpy as np
from scipy import stats, linalg
from sklearn.covariance import GraphLassoCV, LedoitWolf, OAS, \
                               ShrunkCovariance

from sklearn.datasets.base import Bunch
from sklearn.base import BaseEstimator, TransformerMixin
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker, NiftiMasker
import nibabel as nib
from joblib import Parallel, delayed, Memory
from nilearn.datasets import fetch_msdl_atlas
from fetch_data import set_cache_base_dir
from embedding import CovEmbedding, vec_to_sym


CACHE_DIR = set_cache_base_dir()

def fetch_atlas(atlas_name):
    """Retruns selected atlas path
    """
    if atlas_name == 'msdl':
        atlas = fetch_msdl_atlas()['maps']
    elif atlas_name == 'harvard_oxford':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'HarvardOxford-cortl-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'juelich':
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
    
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients 
    between pairs of variables in C, controlling 
    for the remaining variables in C.


    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables.
        Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation
        of C[:, i] and C[:, j] controlling
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


def do_mask_img(func, masker):
    return masker.fit_transform(func)



def compute_connectivity_voxel(roi, voxel, conn):
    """ Returns connectivity of one voxel for a given roi
    """
    
    if conn == 'gl':
        fc = GraphLassoCV(max_iter=1000)
    elif conn == 'lw':
        fc = LedoitWolf()
    elif conn == 'oas':
        fc = OAS()
    elif conn == 'scov':
        fc = ShrunkCovariance()
        
    ts = np.array([roi, voxel]).T

    if conn == 'corr' or conn == 'pcorr':
        cov = np.corrcoef(ts)[0, 1]
    else:
        fc.fit(ts)
        cov = fc.covariance_[0, 0]
    
    return cov


def compute_connectivity_subject(func, masker_ROI, masker_vox, metric):
    """ 
        For each subject :
        1. mask rois and voxels
        2. compute connectivities
    """
    
    print func
    ts_rois = masker_ROI.fit_transform(func)
    ts_vox = masker_vox.fit_transform(func)
    
    fc_ = np.empty((ts_rois.shape[1], ts_vox.shape[1]))
    for i, roi in enumerate(ts_rois.T):
        print 'ROI ', i
        for j, vox in enumerate(ts_vox.T):
            fc_[i, j] = compute_connectivity_voxel(roi, vox, metric)
    return fc_


def compute_connectivity_subjects(imgs, n_jobs, masker_ROI, masker_vox, metric):
    """ All subjects
    """
    return Parallel(n_jobs=n_jobs, verbose=5)(delayed(compute_connectivity_subject)\
                    (img, masker_ROI, masker_vox, metric) for img in imgs)


class ROItoVoxConnectivity(BaseEstimator, TransformerMixin):
    """ ROI to Voxel Connectivity Estimator
    computes the functional connectivity of a list of 4D niimgs,
    according to ROIs defined on an atlas.
    First, the timeseries on ROIs are extracted.
    Then, the connectivity is computed for each pair of ROIs.
    The result is a ravel of half symmetric matrix.
    
    Parameters
    ----------
    atlas : atlas filepath
    metric : metric name (gl, lw, oas, scov, corr, pcorr)
    mask : mask filepath
    detrend : masker param
    low_pass: masker param
    high_pass : masker param
    t_r : masker param
    smoothing : masker param
    resampling_target : masker param
    memory : masker param
    memory_level : masker param
    n_jobs : masker param
    
    Attributes
    ----------
    fc_ : functional connectivity (covariance and precision)
    """
    
    def __init__(self, atlas_name, metric, mask, detrend=True,
                 low_pass=.1, high_pass=.01, t_r=3.,
                 resampling_target='data', smoothing_fwhm=6.,
                 memory='', memory_level=2, n_jobs=1):
        """ - Setting attributes
            - Preparing maskers
        """
        self.atlas = fetch_atlas(atlas_name)
        self.metric = metric
        self.mask = mask
        self.n_jobs = n_jobs
        self.masker_vox = NiftiMasker(mask_img=self.mask,
                                       detrend=detrend,
                                       low_pass=low_pass,
                                       high_pass=high_pass,
                                       t_r=t_r,
                                       smoothing_fwhm=smoothing_fwhm,
                                       memory=memory,
                                       memory_level=memory_level)                                      
        if len(nib.load(self.atlas).shape) == 4:
            self.masker_ROI  = NiftiMapsMasker(maps_img=self.atlas,
                                               mask_img=self.mask,
                                               detrend=detrend,
                                               low_pass=low_pass,
                                               high_pass=high_pass,
                                               t_r=t_r,
                                               resampling_target=resampling_target,
                                               smoothing_fwhm=smoothing_fwhm,
                                               memory=memory,
                                               memory_level=memory_level)
        else:
            self.masker_ROI = NiftiLabelsMasker(labels_img=self.atlas,
                                                 mask_img=self.mask,
                                                 detrend=detrend,
                                                 low_pass=low_pass,
                                                 high_pass=high_pass,
                                                 t_r=t_r,
                                                 resampling_target=resampling_target,
                                                 smoothing_fwhm=smoothing_fwhm,
                                                 memory=memory,
                                                 memory_level=memory_level)

        
    def fit_subjects(self, imgs):
        """ All subjects
        """
        joblibMemory = Memory(CACHE_DIR, mmap_mode='r+', verbose=5)
        return np.asarray(joblibMemory.cache(compute_connectivity_subjects)\
        (imgs, self.n_jobs, self.masker_ROI, self.masker_vox, self.metric))
    
    def fit2(self, imgs):
        """ compute connectivities
        """
        if self.metric == 'correlation' or \
           self.metric == 'partial correlation' or \
           self.metric == 'tangent' :
           ts = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                         do_mask_img)(func, self.masker) for func in imgs)
           cov_embedding = CovEmbedding( kind=self.metric )
           p = np.asarray(vec_to_sym(cov_embedding.fit_transform(ts)))
           ind = np.tril_indices(p.shape[1], k=-1)
           
           self.fc_ = np.asarray([p[i, ...][ind] for i in range(p.shape[0])])
        else:
            p = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                 compute_connectivity_subject)(self.metric, func,
                                    self.masker) for func in imgs)
            self.fc_ = np.asarray(p)[:, 0, :]
        return self.fc_