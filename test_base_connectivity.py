# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:59:34 2015

@author: mehdi.rahim@cea.fr
"""

from base_connectivity import Connectivity, fetch_atlas
import os
import numpy as np
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, set_cache_base_dir

CACHE_DIR = set_cache_base_dir()
dataset = fetch_adni_baseline_rs_fmri()
mask = fetch_adni_masks()['mask_petmr']

conn = Connectivity('msdl', 'tangent', mask, memory=CACHE_DIR, n_jobs=2)
fc = conn.fit(dataset.func[:10])

