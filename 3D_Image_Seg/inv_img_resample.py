import SimpleITK as sitk
import os
import nibabel as nib
from scipy.ndimage import zoom
import numpy as np
#
save_path = './label'
if not os.path.exists(save_path):
    os.mkdir(save_path)

input_dir = './resample_label/'
list_case = [k.split('_')[1] for k in os.listdir(input_dir)]

for i in list_case:
    print(i)
    label_nib = nib.load(os.path.join(input_dir,'crossmoda_'+ i+'_Label', 'crossmoda_'+ i+'_Label.nii.gz'))
    img_name = 'crossmoda_'+i+'_hrT2.nii.gz'
    img = nib.load(
        os.path.join('./resample', img_name))
    header_img = img.header
    label = label_nib.get_fdata()
    ref_img = nib.load(os.path.join('/input', img_name))
    ref_header = ref_img.header

    space_ref = ref_header.get_zooms()
    space_img = header_img.get_zooms()
    size = header_img.get_data_shape()
    new_size = [int(round(size[0] * space_img[0] / space_ref[0])),
                int(round(size[1] * space_img[1] / space_ref[1])),
                int(round(size[2] * space_img[2] / space_ref[2]))]

    label = zoom(label,(1.0*new_size[0]/size[0], 1.0*new_size[1]/size[1], 1.0*new_size[2]/size[2]), order=0 ,mode = 'nearest')
    label = label.astype(np.int16)


    label_r = nib.Nifti1Image(label, ref_img.affine)
    new_img = label_r.__class__(label_r.dataobj[:], ref_img.affine, ref_img.header)
    label_name = 'crossmoda_'+ i+'_Label.nii.gz'
    nib.save(new_img, os.path.join(save_path, label_name))










