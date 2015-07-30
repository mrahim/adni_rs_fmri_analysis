# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:24:36 2015

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
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
import nibabel as nib
from joblib import Parallel, delayed
from nilearn.datasets import fetch_msdl_atlas
from fetch_data import set_cache_base_dir
from embedding import CovEmbedding, vec_to_sym
from nilearn.image import index_img

CACHE_DIR = set_cache_base_dir()


def atlas_rois_to_coords(atlas_name, rois):
    """Returns coords of atlas ROIs
    """

    affine = nib.load(atlas_name).get_affine()
    data = nib.load(atlas_name).get_data()
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

def fetch_dmn_atlas(atlas_name, atlas):
    """ Returns a bunch containing the DMN rois
    and their coordinates
    """

    if atlas_name == 'msdl':
        rois = np.arange(3, 7)
        rois_names = ['L-DMN', 'M-DMN', 'F-DMN', 'R-DMN']
    elif atlas_name == 'mayo':
        rois = np.concatenate((range(39, 43), range(47, 51),
                               range(52, 56), range(62, 68)))
        rois_names = ['adDMN_L', 'adDMN_R', 'avDMN_L', 'avDMN_R', 'dDMN_L_Lat',
                      'dDMN_L_Med', 'dDMN_R_Lat', 'dDMN_R_Med', 'pDMN_L_Lat',
                      'pDMN_L_Med', 'pDMN_R_Lat', 'pDMN_R_Med', 'tDMN_L',
                      'tDMN_R', 'vDMN_L_Lat', 'vDMN_L_Med', 'vDMN_R_Lat',
                      'vDMN_R_Med']
    elif atlas_name == 'canica':
        rois = np.concatenate((range(20, 23), [36]))
        rois_names = ['DMN']*4
    n_rois = len(rois)
    centroids = atlas_rois_to_coords(atlas, rois)

    return Bunch(n_rois=n_rois, rois=rois, rois_names=rois_names,
                 rois_centroids=centroids)

def nii_shape(img):
    """ Returns the img shape
    """

    if isinstance(img, nib.Nifti1Image):
        return img.shape
    else:
        return nib.load(img).shape

def fetch_atlas(atlas_name, rois=False):
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

    dmn = None
    if (atlas_name in ['msdl', 'mayo', 'canica']) and rois:
        dmn = fetch_dmn_atlas(atlas_name, atlas)
        atlas_img = index_img(atlas, dmn['rois'])
        atlas = os.path.join(CACHE_DIR, 'atlas', 'atlas_dmn.nii.gz')
        atlas_img.to_filename(atlas)
    return atlas, dmn

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

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def do_mask_img(masker, func, confound=None):
    """ Masking functional acquisitions
    """
    c = None
    if not confound is None:
        c = np.loadtxt(confound)
    return masker.transform(func, c)

def compute_connectivity_subject(conn, masker, func, confound=None):
    """ Returns connectivity of one fMRI for a given atlas
    """

    ts = do_mask_img(masker, func, confound)

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


class Connectivity(BaseEstimator, TransformerMixin):
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

    def __init__(self, atlas_name, metric, mask, rois=False, detrend=True,
                 low_pass=.1, high_pass=.01, t_r=3.,
                 resampling_target='data', smoothing_fwhm=6.,
                 memory='', memory_level=2, n_jobs=1):

        self.fc_ = None
        self.atlas, self.rois = fetch_atlas(atlas_name, rois)
        self.metric = metric
        self.mask = mask
        self.n_jobs = n_jobs
        if len(nii_shape(self.atlas)) == 4:
            self.masker = NiftiMapsMasker(maps_img=self.atlas,
                                          mask_img=self.mask,
                                          detrend=detrend,
                                          low_pass=low_pass,
                                          high_pass=high_pass,
                                          t_r=t_r,
                                          resampling_target=resampling_target,
                                          smoothing_fwhm=smoothing_fwhm,
                                          memory=memory,
                                          memory_level=memory_level,
                                          verbose=5)
        else:
            self.masker = NiftiLabelsMasker(labels_img=self.atlas,
                                            mask_img=self.mask,
                                            detrend=detrend,
                                            low_pass=low_pass,
                                            high_pass=high_pass,
                                            t_r=t_r,
                                            resampling_target=resampling_target,
                                            smoothing_fwhm=smoothing_fwhm,
                                            memory=memory,
                                            memory_level=memory_level,
                                            verbose=5)

    def fit(self, imgs, confounds=None):
        """ compute connectivities
        """

        self.masker.fit()
        if self.metric == 'correlation' or \
           self.metric == 'partial correlation' or \
           self.metric == 'tangent':

            if confounds is None:
                ts = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                do_mask_img)(self.masker, func) for func in imgs)
            else:
                ts = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                do_mask_img)(self.masker, func, confound)
                for func, confound in zip(imgs, confounds))

            cov_embedding = CovEmbedding(kind=self.metric)
            p_ = np.asarray(vec_to_sym(cov_embedding.fit_transform(ts)))
            ind = np.tril_indices(p_.shape[1], k=-1)

            self.fc_ = np.asarray([p_[i, ...][ind] for i in range(p_.shape[0])])
        else:
            p_ = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
            compute_connectivity_subject)(self.metric,
            self.masker, func, confound)
            for func, confound in zip(imgs, confounds))

            self.fc_ = np.asarray(p_)[:, 0, :]
        return self.fc_
