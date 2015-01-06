"""
    ADNI EPI Quality-check
"""
import os, glob
import pandas as pd
from nilearn.plotting import plot_epi
from nilearn.image import mean_img

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_rs_fmri'
DST_DIR = '/disk4t/mehdi/data/tmp/quality_check'

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))

for idx, row in data.iterrows():
    fmri_file = glob.glob(os.path.join(BASE_DIR,
                                      'I' + str(row.Image_ID), '*.nii'))

    mean_fmri = mean_img(fmri_file[0], n_jobs=4)
    plot_epi(mean_fmri,
             output_file=os.path.join(DST_DIR, row['Subject_ID']),
             title=row['Subject_ID'])
