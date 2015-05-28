# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:52:46 2015

@author: mehdi.rahim@cea.fr
"""

import numpy as np
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, \
                        fetch_adni_longitudinal_rs_fmri
from base_connectivity_classifier import ConnectivityClassifier

#dataset = fetch_adni_baseline_rs_fmri()
dataset = fetch_adni_longitudinal_rs_fmri()
mask = fetch_adni_masks()['mask_petmr']

connclassif = ConnectivityClassifier(dataset.func, mask, atlas='msdl',
                                     dx_group=dataset.dx_group, n_jobs=20)
connclassif.compute_connectivity('corr')
score = connclassif.classify(groups=['AD', 'Normal'])

np.save('scores_longitudinal', score.scores_)
