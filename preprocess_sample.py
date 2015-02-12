"""
    Preprocessing ADNI baseline resting state fmri 
        - Alignement
        - Coreg MRI
        - Normalization MNI
        - Smoothing
"""
import os, glob
import nibabel as nib
from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc


BASE_DIR = '/home/mr243268/data/adni_sample'

###
def delete_scans_fmri(niimg):
    """ Load a 4D nii image and delete 3 first scans
    """
    img = nib.load(niimg)
    img = nib.Nifti1Image(img.get_data()[...,3:], img.get_affine())
    head, tail = os.path.split(niimg)
    img.to_filename(os.path.join(head, 'z' + tail))

###
def delete_scans_adni():
    """ Process adni rs fmri folder
    """
    fmri_paths = sorted(glob.glob(os.path.join(BASE_DIR, 's*')))
    for fmri_path in fmri_paths:
        print os.path.split(fmri_path)[1]
        fmri_file = glob.glob(os.path.join(fmri_path, 'func', 'A*.nii'))
        if len(fmri_file) > 0:
            delete_scans_fmri(fmri_file[0])

###
def clean_dirs_adni():
    """ Remove all processed data
    """
    fmri_paths = sorted(glob.glob(os.path.join(BASE_DIR, 's*')))
    for fmri_path in fmri_paths:
        fmri_file = glob.glob(os.path.join(fmri_path, 'func', 'z*.nii'))
        for f in fmri_file:
            os.remove(f)
            print f, 'removed'
        
"""
# Delete first 3 scans
clean_dirs_adni()
delete_scans_adni()
"""

jobfile = 'preprocess_sample_config.ini'
dataset_dir = BASE_DIR

# sourcing FSL
os.system('source /etc/fsl/4.1/fsl.sh')

# preprocess the data
results = do_subjects_preproc(jobfile, dataset_dir=BASE_DIR)