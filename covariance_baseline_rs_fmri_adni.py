"""
    Covariance matrix rs fmri ADNI
"""

import os
from fetch_data.datasets import fetch_adni_rs_fmri
from nilearn.datasets import fetch_msdl_atlas
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMapsMasker
from nilearn.plotting import plot_roi, plot_stat_map, plot_img
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.covariance import EmpiricalCovariance

###############################################################################
###############################################################################
plotted_subject = 0  # subject to plot


FIG_PATH = '/disk4t/mehdi/data/tmp/figures'

import matplotlib.pyplot as plt
import matplotlib
# Copied from matplotlib 1.2.0 for matplotlib 0.99 compatibility.
_bwr_data = ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
plt.cm.register_cmap(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bwr", _bwr_data))

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
    plt.xticks(np.linspace(0.4,1.0,7), np.arange(40,110,10), fontsize=18)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join(['boxplot_adni_baseline_rs_fmri',
                          ext])
        plt.savefig(os.path.join(FIG_PATH, fname), transparent=True)


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    prec = prec.copy()  # avoid side effects

    # Display sparsity pattern
    sparsity = prec == 0
    plt.figure()
    plt.imshow(sparsity, interpolation="nearest")
    plt.title("%s / sparsity" % title)

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plt.figure()
    plt.imshow(cov, interpolation="nearest",
              vmin=-1, vmax=1, cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    plt.title("%s / covariance" % title)

    # Display precision matrix
    plt.figure()
    plt.imshow(prec, interpolation="nearest",
              vmin=-span, vmax=span,
              cmap=plt.cm.get_cmap("bwr"))
    plt.colorbar()
    plt.title("%s / precision" % title)

###############################################################################
###############################################################################

CACHE_DIR = os.path.join('/', 'disk4t', 'mehdi',
                         'data', 'tmp')

dataset = fetch_adni_rs_fmri()
atlas = fetch_msdl_atlas()
func_files = dataset['func']
dx_group = np.array(dataset['dx_group'])
idx = {}
for g in ['AD', 'LMCI', 'EMCI', 'Normal']:
    idx[g] = np.where(dx_group == g)

atlas4d = nib.load(atlas['maps'])
atlas4d_data = atlas4d.get_data()
atlas3d_data = np.sum(atlas4d_data, axis=3)
atlas3d = nib.Nifti1Image(atlas3d_data, atlas4d.get_affine())

n_subjects = len(func_files)
subjects = []
cov_feat = []

for subject_n in range(n_subjects):
    filename = func_files[subject_n]
    print("Processing file %s" % filename)
    print("-- Computing region signals ...")
    masker = NiftiMapsMasker(atlas["maps"],
                             resampling_target="maps", standardize=False,
                             memory=CACHE_DIR, memory_level=1, verbose=0)
    region_ts = masker.fit_transform(filename)
    subjects.append(region_ts)
    print("-- Computing covariances")
    cov_matrix = np.cov(region_ts.T)
    cov_feat.append(cov_matrix[np.tril_indices(len(cov_matrix))])

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit
    
cov_feat = np.array(cov_feat)

nb_iter = 100
pg_counter = 0
groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]
score = np.zeros((nb_iter, len(groups)))
for gr in groups:
    g1_feat = cov_feat[idx[gr[0]][0]]
    g2_feat = cov_feat[idx[gr[1]][0]]
    x = np.concatenate((g1_feat, g2_feat), axis=0)
    y = np.ones(len(x))
    y[len(x) - len(g2_feat):] = 0
    
    estim = SVC(kernel='linear')
    sss = StratifiedShuffleSplit(y, n_iter=nb_iter, test_size=0.2)
    # 1000 runs with randoms 80% / 20% : StratifiedShuffleSplit
    counter = 0
    for train, test in sss:
        Xtrain, Xtest = x[train], x[test]
        Ytrain, Ytest = y[train], y[test]
        Yscore = estim.fit(Xtrain,Ytrain)
        score[counter, pg_counter] = estim.score(Xtest, Ytest)
        counter += 1
    pg_counter += 1

plot_shufflesplit(score, groups)