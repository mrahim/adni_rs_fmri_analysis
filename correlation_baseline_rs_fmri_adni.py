# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 09:35:32 2015

@author: mr243268
"""
import os
import numpy as np
from fetch_data import fetch_adni_rs_fmri
from nilearn.datasets import fetch_msdl_atlas
from nilearn.input_data import NiftiMapsMasker
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt


FIG_PATH = '/disk4t/mehdi/data/tmp/figures'

def plot_shufflesplit(score, pairwise_groups):
    bp = plt.boxplot(score, 0, '', 0)
    for key in bp.keys():
        for box in bp[key]:
            box.set(linewidth=2)
    plt.grid(axis='x')
    plt.xlim([.4, 1.])
    plt.xlabel('Accuracy (%)', fontsize=18)
    plt.title('Shuffle Split Accuracies ',
              fontsize=17)
    plt.yticks(range(1,7), ['AD/Normal', 'AD/EMCI', 'AD/LMCI', 'LMCI/Normal', 'LMCI/EMCI', 'EMCI/Normal'], fontsize=18)
    #plt.yticks(range(1,7), ['AD/Normal', 'AD/EMCI', 'AD/LMCI', 'EMCI/LMCI', 'EMCI/Normal', 'LMCI/Normal'], fontsize=18)
    plt.xticks(np.linspace(0.4,1.0,7), np.arange(40,110,10), fontsize=18)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join(['boxplot_adni_baseline_rs_fmri_corr',
                          ext])
        plt.savefig(os.path.join(FIG_PATH, fname), transparent=True)


CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi',
                         'data', 'tmp')
                         
dataset = fetch_adni_rs_fmri()
atlas = fetch_msdl_atlas()
func_files = dataset['func']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

n_subjects = len(func_files)
subjects = []
corr_feat = []
corr_mat = []
for subject_n in range(n_subjects):
    filename = func_files[subject_n]
    print("Processing file %s" % filename)
    print("-- Computing region signals ...")
    masker = NiftiMapsMasker(atlas["maps"],
                             resampling_target="maps", standardize=False,
                             memory=CACHE_DIR, memory_level=1, verbose=0)
    region_ts = masker.fit_transform(filename)
    subjects.append(region_ts)
    print("-- Computing correlations")
    corr_matrix = np.corrcoef(region_ts.T)
    corr_feat.append(corr_matrix[np.tril_indices(len(corr_matrix))])
    corr_mat.append(corr_matrix)

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
    
corr_mat = np.array(corr_mat)
#corr_feat = StandardScaler().fit_transform(np.array(corr_feat))
corr_feat  = np.array(corr_feat)

nb_iter = 100
pg_counter = 0
"""
groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]
"""
groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['LMCI', 'Normal'], ['EMCI', 'LMCI'], ['EMCI', 'Normal']]
score = np.zeros((nb_iter, len(groups)))
for gr in groups:
    g1_feat = corr_feat[idx[gr[0]][0]]
    g2_feat = corr_feat[idx[gr[1]][0]]
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0
    svr = SVC(kernel='linear')
    param_grid = {'C': np.logspace(-3,3,7), 'kernel': ['linear']}
    estim = GridSearchCV(cv=None, estimator=svr,
                         param_grid=param_grid, n_jobs=1)
    sss = StratifiedShuffleSplit(y, n_iter=nb_iter, test_size=0.1)
    # 100 runs with randoms 90% / 10% : StratifiedShuffleSplit
    counter = 0
    for train, test in sss:
        Xtrain, Xtest = x[train], x[test]
        Ytrain, Ytest = y[train], y[test]
        Yscore = estim.fit(Xtrain,Ytrain)
        score[counter, pg_counter] = estim.score(Xtest, Ytest)
        counter += 1
    pg_counter += 1
    
plot_shufflesplit(score, groups)