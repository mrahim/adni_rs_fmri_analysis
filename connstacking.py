# -*- coding: utf-8 -*-
"""

# 1- Compute connectivity features
# 2- Classify for pairwise and multiclass
# 3- Stack prediction

Created on Wed May 27 15:13:23 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks




###############################################################################
# Main loop
###############################################################################

dataset = fetch_adni_baseline_rs_fmri()
mask = fetch_adni_masks()['mask_petmr']



