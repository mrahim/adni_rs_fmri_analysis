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
from sklearn.metrics import accuracy_score
from base_connectivity import Connectivity

CACHE_DIR = set_cache_base_dir()


###############################################################################
# Classification
###############################################################################
def StratifiedSubjectShuffleSplit(dataset, groups, n_iter=100, test_size=.3,
                                  random_state=42):
    """ Stratified ShuffleSplit on subjects
    (train and test size may change depending on the number of acquistions)"""

    idx = set_group_indices(dataset.dx_group)
    groups_idx = np.hstack([idx[group] for group in groups])

    subjects = np.asarray(dataset.subjects)
    subjects = subjects[groups_idx]

    dx = np.asarray(dataset.dx_group)
    dx = dx[groups_idx]

    # extract unique subject ids and dx
    subjects_unique_values, \
    subjects_unique_indices = np.unique(subjects, return_index=True)

    # extract indices for the needed groups
    dx_unique_values = dx[subjects_unique_indices]
    y = dx_unique_values

    # generate folds stratified on dx
    sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size,
                                 random_state=random_state)
    ssss = []
    for tr, ts in sss:
        # get training subjects
        subjects_tr = subjects_unique_values[tr]

        # get testing subjects
        subjects_ts = subjects_unique_values[ts]

        # get all subject indices
        train = []
        test = []
        for subj in subjects_tr:
            train.extend(np.where(subjects == subj)[0])
        for subj in subjects_ts:
            test.extend(np.where(subjects == subj)[0])

        # append ssss
        ssss.append([train, test])
    return ssss


def SubjectShuffleSplit(dataset, groups, n_iter=100,
                        test_size=.3, random_state=42):
    """ Specific ShuffleSplit (train on all subject images,
    but test only on one image of the remaining subjects)"""

    idx = set_group_indices(dataset.dx_group)
    groups_idx = np.hstack([idx[group] for group in groups])

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
        test_set = np.array([], dtype=int)
        for subj in subjects_unique[test]:
            subj_ind = np.where(subjects == subj)
            test_set = np.concatenate((test_set, subj_ind[0]))
        subj_ss.append([train_set, test_set])
    return subj_ss


def train_and_test(classifier, X, y, train, test, subjects=None):
    """ Returns accuracy and coeffs for a train and test
    """
    classifier.fit(X[train, :], y[train])
    score = classifier.score(X[test, :], y[test])
    yp = classifier.predict(X[test, :])
    yd = classifier.decision_function(X[test, :])
    B = Bunch(score=score, coef=classifier.coef_,
              y_dec=yd,
              y=y[test],
              y_pred=yp,
              subj=subjects[test])
    return B


def average_predictions(y, decision_function, subjects):
    """ Returns a score averaged on subject acquisitions
    """
    # Compute subj average decision_function
    subjects_unique_values, \
    subjects_unique_indices = np.unique(subjects, return_index=True)

    decision_function_unique = [np.mean(decision_function[subjects == subj])
                                for subj in subjects_unique_values]
    decision_function_unique = np.array(decision_function_unique)

    # Threshold for the prediction
    yp = np.zeros(decision_function_unique.shape, dtype=np.int)
    yp[decision_function_unique > 0] = 1

    # Compute accuracy
    score = accuracy_score(y[subjects_unique_indices], yp)
    return (decision_function_unique, yp, y[subjects_unique_indices], y,
            subjects_unique_indices, score)


def classify_connectivity(X, y, sss, classifier_name, n_jobs=-1,
                          subjects=None):
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

    p = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(train_and_test)(classifier, X, y, train, test, subjects)
        for train, test in sss)
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

    def __init__(self, imgs, mask, atlas, dx_group, rois=False, n_iter=100,
                 memory=CACHE_DIR, memory_level=2, n_jobs=-1, random_state=42):
        """ initialization
        """
        self.imgs = np.array(imgs)
        self.mask = mask
        self.atlas = atlas
        self.rois = rois
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.memory = memory
        self.memory_level = memory_level
        self.idx = set_group_indices(dx_group)
        self.random_state = random_state

    def compute_connectivity(self, metric, confounds=None):
        """ Return covariance matrix
        """
        conn = Connectivity(self.atlas, metric, self.mask, self.rois,
                            memory=CACHE_DIR, n_jobs=self.n_jobs,
                            smoothing_fwhm=None)
        self.connectivity = conn.fit(self.imgs, confounds)

    def classify(self, dataset=None, groups=['AD', 'MCI'],
                 classifier_name='logreg_l2'):
        """ Returns accuracy scores
        """
        if hasattr(self, 'connectivity'):
            groups_idx = np.hstack([self.idx[group] for group in groups])
            subjects = np.array(dataset.subjects)
            subjects = subjects[groups_idx]
            y = np.hstack([[i] * len(self.idx[group])
                          for i, group in enumerate(groups)])

            X = self.connectivity[groups_idx, :]

            if dataset is None:
                sss = StratifiedShuffleSplit(y, n_iter=self.n_iter,
                                             test_size=.25,
                                             random_state=self.random_state)
            else:
                sss = StratifiedSubjectShuffleSplit(dataset, groups,
                                                n_iter=self.n_iter,
                                                test_size=.3,
                                                random_state=self.random_state)

#                sss = SubjectShuffleSplit(dataset, groups, n_iter=self.n_iter,
#                                          test_size=.2,
#                                          random_state=self.random_state)

            results = classify_connectivity(X, y, sss, classifier_name,
                                            n_jobs=self.n_jobs,
                                            subjects=subjects)

            self.y_pred_ = map(lambda r: r['y_pred'], results)
            self.y_dec_ = map(lambda r: r['y_dec'], results)
            self.y_ = map(lambda r: r['y'], results)
            self.coefs_ = map(lambda r: r['coef'], results)
            self.scores_ = map(lambda r: r['score'], results)
            self.subj_ = map(lambda r: r['subj'], results)

            self.results_ = map(average_predictions, self.y_, self.y_dec_,
                                self.subj_)

            self.scores_ = np.asarray(self.scores_)
            self.coefs_ = np.asarray(self.coefs_)
            self.subj_ = np.asarray(self.subj_)
        else:
            raise ValueError('Connectivity not yet computed !')
