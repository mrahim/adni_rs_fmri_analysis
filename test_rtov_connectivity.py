# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:03:25 2015

@author: mehdi.rahim@cea.fr
"""

from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, set_cache_base_dir
from base_rtov_connectivity import ROItoVoxConnectivity

CACHE_DIR = set_cache_base_dir()
mask = fetch_adni_masks()['mask_fmri']
dataset = fetch_adni_baseline_rs_fmri()

conn = ROItoVoxConnectivity(atlas_name='msdl',
                            metric='corr',
                            mask=mask,
                            memory=CACHE_DIR,
                            memory_level=2,
                            n_jobs=10)

output = conn.fit_subjects(dataset.func[:10])


