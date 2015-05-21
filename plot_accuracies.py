# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:31:56 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

###############################################################################
# Retrieve scores
###############################################################################
def retrieve_scores(atlas_name, metric, classifier_name, group):
    ''' boxplot accuracies
    '''
    
    fname = '_'.join([group[0], group[1], atlas_name, metric, classifier_name])
    input_file = os.path.join(BASE_DIR, fname + '.npz')

    if not os.path.isfile(input_file):
        return 0
    else:
        data = np.load(input_file)['data']
        accuracies = []
        for d in data:
            accuracies.append(d.score)

    accuracies = np.asarray(accuracies)
    if np.median(accuracies) < .5:
        accuracies = 1 - accuracies
    return accuracies

###############################################################################
# FACETGRID plotting
###############################################################################
def plot_facetgrid():
    ''' FacetGrid (Metric, Classifier)
    '''
    dataframe = pd.DataFrame(columns=['accuracy', 'metric', 'classifier', 'group'])
    for group in groups:
        for metric in metrics:
            for classifier_name in classifier_names:
                data = retrieve_scores(atlas_name, metric, classifier_name, group)
                dictionary = {}
                dictionary['accuracy'] = data
                dictionary['metric'] = metric
                dictionary['classifier'] = classifier_name
                dictionary['group'] = '_'.join(group)
                df = pd.DataFrame.from_dict(dictionary)
                dataframe = pd.concat([dataframe, df])
    
    g = sns.FacetGrid(dataframe, row='classifier', col='metric', margin_titles=True)
    g.map(sns.barplot, 'group', 'accuracy')
    g.savefig('msdl_accuracies.pdf')
    g.savefig('msdl_accuracies.png')


###############################################################################
# BOXPLOT plotting
###############################################################################
def boxplot_grid(atlas_name):
    """ Subplots accuracies according to metrics and classfiers.
    for a given atlas
    """
    plt.figure(figsize=(22, 18))
    for i, metric in enumerate(metrics):
        for j, classifier_name in enumerate(classifier_names):
            print i, j, len(metrics), len(classifier_names), (j+1) + i*len(classifier_names)
            data = []
            group_names = []
            for group in groups:
                data.append(retrieve_scores(atlas_name, metric, classifier_name, group))
                group_names.append('/'.join(group))
            plt.subplot(len(metrics), len(classifier_names),
                        (j + 1) + i * len(classifier_names))
            ax = sns.boxplot(data, names=group_names)
            plt.ylim([.3, .9])
            plt.title(classifier_names_full[classifier_name], fontsize=16)
        plt.ylabel(metrics_full[metric], fontsize=16)
        ax.yaxis.set_label_position('right')
    plt.suptitle(atlas_names_full[atlas_name], fontsize=22, y=.95)
    plt.savefig(atlas_name + '_grid_metric_classifier.pdf')
    plt.savefig(atlas_name + '_grid_metric_classifier.png')

###############################################################################
# Main loop
###############################################################################

atlas_names = ['canica', 'msdl', 'mayo', 'harvard_oxford', 'juelich']
atlas_names_full = {'msdl' : 'Atlas MSDL',
                    'mayo' : 'Atlas Mayo Clinic',
                    'harvard_oxford': 'Atlas Harvard-Oxford',
                    'juelich': 'Atlas Juelich',
                    'canica': 'CanICA 61 ROIs'}
                    
atlas_names = ['canica', 'mayo', 'harvard_oxford', 'juelich', 'msdl']
                    
metrics = ['gl', 'lw', 'oas', 'scov', 'corr', 'pcorr']
metrics_full = {'gl' : 'GraphLasso',
                'lw' : 'LedoitWolf',
                'oas': 'OraclApproxShrink',
                'scov' : 'ShrunkCov',
                'corr' : 'Pearson Corr',
                'pcorr' : 'Partial Corr'}
classifier_names = ['logreg_l1', 'logreg_l2', 'ridge', 'svc_l1', 'svc_l2']
classifier_names_full = {'logreg_l1' : 'LogisticRegression_l1',
                         'logreg_l2' : 'LogisticRegression_l2',
                         'ridge' : 'RidgeClassifier',
                         'svc_l1' : 'SVC_l1',
                         'svc_l2' : 'SVC_l2'}
groups = [['AD', 'MCI'], ['AD', 'Normal'], ['MCI', 'Normal']]

for atlas_name in atlas_names:
    print atlas_name
    BASE_DIR = '/disk4t/mehdi/data/tmp/conn_' + atlas_name
    boxplot_grid(atlas_name)
    