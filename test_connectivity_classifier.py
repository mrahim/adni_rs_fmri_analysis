# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:52:46 2015

@author: mehdi.rahim@cea.fr
"""

from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks
from base_connectivity_classifier import ConnectivityClassifier

dataset = fetch_adni_baseline_rs_fmri()
mask = fetch_adni_masks()['mask_petmr']

connclassif = ConnectivityClassifier(dataset.func, mask, atlas='msdl',
                                     dx_group=dataset.dx_group, n_jobs=20)
connclassif.compute_connectivity('corr')
connclassif.classify(groups=['AD', 'Normal'])