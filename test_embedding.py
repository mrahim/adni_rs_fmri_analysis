# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:21:42 2015

@author: mr243268
"""

from embedding import CovEmbedding, vec_to_sym
from nilearn.datasets import fetch_nyu_rest, fetch_msdl_atlas
from nilearn.input_data import NiftiMapsMasker

dataset = fetch_nyu_rest(n_subjects=1)
atlas = fetch_msdl_atlas()

masker = NiftiMapsMasker(atlas['maps'], detrend=True, standardize=True)
masker.fit()
ts = masker.transform(dataset.func[0])
cov_embed = CovEmbedding(kind='tangent')
output = cov_embed.fit_transform([ts])

m = vec_to_sym(output)
