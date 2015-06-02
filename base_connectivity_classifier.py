# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:24:06 2015

@author: mehdi.rahim@cea.fr
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.datasets.base import Bunch
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from fetch_data import set_group_indices, set_cache_base_dir
from sklearn.base import BaseEstimator, TransformerMixin
from base_connectivity import Connectivity

CACHE_DIR = set_cache_base_dir()

###############################################################################
# Classification
###############################################################################
def SubjectShuffleSplit(dataset, groups, n_iter=100,
                        test_size=.3, random_state=42):
    """ Specific ShuffleSplit (train on all subject images,
    but test only on one image of the remaining subjects)"""

    idx = set_group_indices(dataset.dx_group)
    groups_idx = np.hstack((idx[groups[0]],
                            idx[groups[1]]))

    subjects = np.asarray(dataset.subjects)
    subjects = subjects[groups_idx]
    subjects_unique = np.unique(subjects)

    n = len(subjects_unique)
    ss = ShuffleSplit(n, n_iter=n_iter,
                      test_size=test_size, random_state=random_state)

    subj_ss = []
    for train, test in ss:
        train_set = np.array([], dtype=int)
        for subj in subjects_unique[train]:
            subj_ind = np.where(subjects == subj)
            train_set = np.concatenate((train_set, subj_ind[0]))
        test_set = []
        for subj in subjects_unique[test]:
            subj_ind = np.where(subjects == subj)
            test_set.append(subj_ind[0][0])
        test_set = np.asarray(test_set)
        subj_ss.append([train_set, test_set])

    return subj_ss



def train_and_test(classifier, X, y, train, test):
    """ Returns accuracy and coeffs for a train and test
    """
    classifier.fit(X[train, :], y[train])
    score = classifier.score(X[test, :], y[test])
    B = Bunch(score=score, coef=classifier.coef_)
    return B

def classify_connectivity(X, y, sss, classifier_name, n_jobs=-1):
    """ Returns 100 shuffle split scores
    """
    if classifier_name == 'logreg_l1':
        classifier = LogisticRegression(penalty='l1', dual=False,
                                        random_state=42)
    elif classifier_name == 'logreg_l2':
        classifier = LogisticRegression(penalty='l2', random_state=42)
    elif classifier_name == 'ridge':
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))
    elif classifier_name == 'svc_l2':
        classifier = LinearSVC(penalty='l2', random_state=42)
    elif classifier_name == 'svc_l1':
        classifier = LinearSVC(penalty='l1', dual=False, random_state=42)

    p = Parallel(n_jobs=n_jobs, verbose=5)(delayed(train_and_test)(
                 classifier, X, y, train, test) for train, test in sss)
    return np.asarray(p)

class ConnectivityClassifier(BaseEstimator, TransformerMixin):
    """ Connectivity based binary classification
    Parameters
    ----------        
    Attributes
    ----------
    scores_
    coefs_
    """
    
    def __init__(self, imgs, mask, atlas, dx_group, n_iter=100,
                 memory=CACHE_DIR, memory_level=2, n_jobs=-1, random_state=42):
        """ initialization
        """
        self.imgs = imgs
        self.mask = mask
        self.atlas = atlas
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.memory = memory
        self.memory_level = memory_level
        self.idx = set_group_indices(dx_group)
        self.random_state = random_state
        
                                 
    def compute_connectivity(self, metric):
        """ Return covariance matrix
        """
        conn = Connectivity(self.atlas, metric, self.mask,
                            memory=CACHE_DIR, n_jobs=self.n_jobs)
        self.connectivity = conn.fit(self.imgs)
        
    
    def classify(self, dataset=None, groups=['AD', 'MCI'], classifier_name='logreg_l2'):
        """ Returns accuracy scores
        """
        if hasattr(self, 'connectivity'):        
            groups_idx = np.hstack((self.idx[groups[0]],
                                    self.idx[groups[1]]))
            X = self.connectivity[groups_idx, :]
            y = np.asarray([+1] * len(self.idx[groups[0]]) +
                           [-1] * len(self.idx[groups[1]]))
                           
            if dataset is None:
                sss = StratifiedShuffleSplit(y, n_iter=self.n_iter,
                                             test_size=.25,
                                             random_state=self.random_state)
            else:
                sss = SubjectShuffleSplit(dataset, groups, n_iter=self.n_iter,
                                          test_size=.3,
                                          random_state=self.random_state)
            results = classify_connectivity(X, y, sss, classifier_name,
                                            n_jobs=self.n_jobs)
            self.scores_ = []
            self.coefs_ = []
            for r in results:
                self.scores_.append(r['score'])
                self.coefs_.append(r['coef'])
            self.scores_ = np.asarray(self.scores_)
            self.coefs_ = np.asarray(self.coefs_)
        else:
            raise ValueError('Connectivity not yet computed !')