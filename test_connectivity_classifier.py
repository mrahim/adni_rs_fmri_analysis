# -*- coding: utf-8 -*-
"""
Created on Thu May 28 10:52:46 2015

@author: mehdi.rahim@cea.fr
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from fetch_data import fetch_adni_baseline_rs_fmri, fetch_adni_masks, \
                        fetch_adni_longitudinal_rs_fmri_DARTEL, set_cache_base_dir
from base_connectivity_classifier import ConnectivityClassifier
from sklearn.metrics import confusion_matrix

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
#sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)


CACHE_DIR = set_cache_base_dir()

#dataset = fetch_adni_baseline_rs_fmri()
dataset = fetch_adni_longitudinal_rs_fmri_DARTEL()
mask = fetch_adni_masks()['mask_fmri']


all_groups = [['AD', 'MCI'], ['AD', 'Normal'], ['MCI', 'Normal']]

atlas_names = ['msdl', 'canica141', 'canica', 'mayo',
               'harvard_oxford', 'juelich', 'tvmsdl']
               
classifier_names = ['ridge', 'svc_l1', 'svc_l2',
                    'logreg_l1', 'logreg_l2']             
    
conn_names = ['corr', 'correlation', 'tangent', 'gl', 'lw', 'oas', 'scov']

###

atlas_names = ['mayo']
classifier_names = ['logreg_l2']            
conn_names = ['corr']
#all_groups = [['AD', 'MCI', 'Normal']]
#all_groups = [['AD', 'MCI'], ['AD', 'Normal'], ['MCI', 'Normal']]
all_groups = [['AD', 'MCI']]

for atlas_name in atlas_names:
    print atlas_name
    for conn_name in conn_names:
        connclassif = ConnectivityClassifier(dataset.func, mask,
                                             atlas=atlas_name,
                                             dx_group=dataset.dx_group,
                                             n_jobs=20)
        connclassif.compute_connectivity(conn_name, confounds=dataset.motions)
        for groups in all_groups:
            print groups
            for classifier_name in classifier_names:
                connclassif.classify(dataset=dataset,
                                     groups=groups,
                                     classifier_name=classifier_name)
                if len(groups)>2:
                    cc = [ confusion_matrix(connclassif.yp_[i][1],
                                            connclassif.yp_[i][0])
                                            for i in range(connclassif.n_iter) ]
                    t = np.sum(cc, axis=0)
                    m = t/(1.*np.sum(t, axis=1)[:, np.newaxis])
                    plt.imshow(m/np.sum(m, axis=1), interpolation='nearest',
                               cmap=plt.cm.gray)
                output_folder = os.path.join(CACHE_DIR, 'DARTEL_SUBJ',
                                             '_'.join(['conn', atlas_name]))
                if not os.path.isdir(output_folder):
                    os.mkdir(output_folder)
                groups_name = '_'.join(groups)
                output_file = os.path.join(output_folder,
                                           '_'.join([groups_name,
                                                     atlas_name, conn_name,
                                                     classifier_name]))
                np.savez_compressed(output_file, data=connclassif.scores_)
    
