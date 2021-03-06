# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:27:14 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from scipy import stats, linalg
from sklearn.covariance import GraphLassoCV, LedoitWolf, OAS, \
                               ShrunkCovariance

from sklearn.datasets.base import Bunch
from sklearn.base import BaseEstimator, TransformerMixin
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
import nibabel as nib
from joblib import Parallel, delayed
from nilearn.datasets import fetch_msdl_atlas
from fetch_data import set_cache_base_dir
from embedding import CovEmbedding, vec_to_sym


CACHE_DIR = set_cache_base_dir()


def atlas_rois_to_coords(atlas_name, rois):
    """Returns coords of atlas ROIs
    """

    atlas = fetch_atlas(atlas_name)
    affine = nib.load(atlas).get_affine()
    data = nib.load(atlas).get_data()
    centroids = []
    if len(data.shape) == 4:
        for i in range(data.shape[-1]):
            voxels = np.where(data[..., i] > 0)
            centroid = np.mean(voxels, axis=1)
            dvoxels = data[..., i]
            dvoxels = dvoxels[voxels]
            voxels = np.asarray(voxels).T
            centroid = np.average(voxels, axis=0, weights=dvoxels)
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

    centroids = np.asarray(centroids)[rois]
    return centroids



def fetch_dmn_atlas(atlas_name):
    """ Returns a bunch containing the DMN rois
    and their coordinates
    """
    
    if atlas_name == 'msdl':    
        rois = np.arange(3, 7)
        rois_names = ['L-DMN', 'M-DMN', 'F-DMN', 'R-DMN']
    elif atlas_name == 'mayo':
        rois = np.concatenate(( range(39, 43), range(47, 51),
                                range(52, 56), range(62, 68) ))
        rois_names = ['adDMN_L', 'adDMN_R', 'avDMN_L', 'avDMN_R', 'dDMN_L_Lat',
                      'dDMN_L_Med', 'dDMN_R_Lat', 'dDMN_R_Med', 'pDMN_L_Lat',
                      'pDMN_L_Med', 'pDMN_R_Lat', 'pDMN_R_Med', 'tDMN_L',
                      'tDMN_R', 'vDMN_L_Lat', 'vDMN_L_Med', 'vDMN_R_Lat',
                      'vDMN_R_Med']
    elif atlas_name == 'canica':
        rois = np.concatenate((range(20, 23), [36]))
        rois_names = ['DMN']*4
    n_rois = len(rois)
    centroids = atlas_rois_to_coords(atlas_name, rois)
    
    return Bunch(n_rois=n_rois, rois=rois, rois_names=rois_names,
                 rois_centroids=centroids)
    

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

def compute_network_connectivity_subject(conn, func, masker, rois):
    """ Returns connectivity of one fMRI for a given atlas
    """
    ts = masker.fit_transform(func)
    ts = np.asarray(ts)[ :, rois]
    
    if conn == 'gl':
        fc = GraphLassoCV(max_iter=1000)
    elif conn == 'lw':
        fc = LedoitWolf()
    elif conn == 'oas':
        fc = OAS()
    elif conn == 'scov':
        fc = ShrunkCovariance()
        
	fc = Bunch(covariance_=0, precision_=0)
    
    if conn == 'corr' or conn == 'pcorr':
        fc = Bunch(covariance_=0, precision_=0)
        fc.covariance_ = np.corrcoef(ts)
        fc.precision_ = partial_corr(ts)
    else:
        fc.fit(ts)
    ind = np.tril_indices(ts.shape[1], k=-1)
    return fc.covariance_[ind], fc.precision_[ind]


    
class NetworkConnectivity(BaseEstimator, TransformerMixin):
    """ Connectivity Estimator
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
    
    def __init__(self, atlas_name, rois, metric, mask, detrend=True,
                 low_pass=.1, high_pass=.01, t_r=3.,
                 resampling_target='data', smoothing_fwhm=6.,
                 memory='', memory_level=2, n_jobs=1):
        self.atlas = fetch_atlas(atlas_name)
        self.rois = np.asarray(rois)
        self.metric = metric
        self.mask = mask
        self.n_jobs = n_jobs
        if len(nib.load(self.atlas).shape) == 4:
            self.masker  = NiftiMapsMasker(maps_img=self.atlas,
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
            self.masker  = NiftiLabelsMasker(labels_img=self.atlas,
                                             mask_img=self.mask,
                                             detrend=detrend,
                                             low_pass=low_pass,
                                             high_pass=high_pass,
                                             t_r=t_r,
                                             resampling_target=resampling_target,
                                             smoothing_fwhm=smoothing_fwhm,
                                             memory=memory,
                                             memory_level=memory_level)

    def fit(self, imgs):
        """ compute connectivities
        """
        if self.metric == 'correlation' or \
           self.metric == 'partial correlation' or \
           self.metric == 'tangent' :
           ts = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                         do_mask_img)(func, self.masker) for func in imgs)
           
           #ts = np.asarray(ts)[0, :, self.rois].T
                       
           cov_embedding = CovEmbedding( kind=self.metric )
           p = np.asarray(vec_to_sym(cov_embedding.fit_transform(ts)))
           ind = np.tril_indices(p.shape[1], k=-1)
           self.fc_ = np.asarray([p[i, ...][ind] for i in range(p.shape[0])])
        else:
            p = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                 compute_network_connectivity_subject)(self.metric, func,
                                    self.masker, self.rois) for func in imgs)
            self.fc_ = np.asarray(p)[:, 0, :]
        return self.fc_