"""
    Preprocessing ADNI baseline resting state fmri 
        - Alignement
        - Coreg MRI
        - Normalization MNI
        - Smoothing
"""
import os
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc


BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_rs_fmri_mri'

jobfile = 'rs_fmri_adni_preproc.ini'
dataset_dir = BASE_DIR


# sourcing FSL
os.system('source /etc/fsl/4.1/fsl.sh')


# preprocess the data
results = do_subjects_preproc(jobfile, dataset_dir=BASE_DIR)