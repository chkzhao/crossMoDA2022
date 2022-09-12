import SimpleITK as sitk
import os

ref_img = sitk.ReadImage('./crossmoda_145_hrT2.nii.gz')
#
save_path = './resample'
if not os.path.exists(save_path):
    os.mkdir(save_path)

input_dir = '../target_validation/'
path_img = os.path.join(input_dir,'crossmoda_{}_hrT2.nii.gz')
list_case = [k.split('_')[1] for k in os.listdir(input_dir)]

for i in list_case:
    img = sitk.ReadImage(os.path.join(input_dir,'crossmoda_'+ i +'_hrT2.nii.gz'))

    space_ref = ref_img.GetSpacing()
    space_img = img.GetSpacing()
    size = img.GetSize()
    new_size = [int(size[0] * space_img[0] / space_ref[0]), int(size[1] * space_img[1] / space_ref[1]),
                int(size[2] * space_img[2] / space_ref[2])]

    origin = img.GetOrigin()

    img_r = sitk.Resample(img, new_size,
                          sitk.Transform(),
                          sitk.sitkLinear,
                          origin,
                          ref_img.GetSpacing(),
                          ref_img.GetDirection(),
                          0,
                          img.GetPixelID())

    spacing_new = img_r.GetSpacing()
    new_size = img_r.GetSize()
    save_name = 'crossmoda_'+ i +'_hrT2.nii.gz'

    sitk.WriteImage(img_r, os.path.join(save_path, save_name))






