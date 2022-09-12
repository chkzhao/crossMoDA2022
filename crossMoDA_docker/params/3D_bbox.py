from skimage.measure import label, regionprops
import numpy as np

def crop_seg(seg, coc_size = [32, 32, 12]):
    lbl_0 = label(seg)
    props = regionprops(lbl_0)
    bbox_lists = []
    for prop in props:
        bbox = prop.bbox

        x0, y0, z0, x1, y1, z1 = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
        x_b = np.random.choice(coc_size[0] + 1 - (x1 - x0), 1).item()
        y_b = np.random.choice(coc_size[1] + 1 - (y1 - y0), 1).item()
        z_b = np.random.choice(coc_size[2] + 1 - (z1 - z0), 1).item()

        x0_b = x0 - x_b
        x1_b = x1 + coc_size[0] - (x1 - x0) - x_b

        y0_b = y0 - y_b
        y1_b = y1 + coc_size[1] - (y1 - y0) - y_b

        z0_b = z0 - z_b
        z1_b = z1 + coc_size[2] - (z1 - z0) - z_b

        bbox_lists.append([x0_b, x1_b, y0_b,y1_b, z0_b,z1_b])

    return bbox_lists

# seg_img = nib.load('/ocean/projects/asc170022p/yanwuxu/crossMoDA/data_resampled/crossmoda_training/source_training/crossmoda_18_Label.nii.gz')
# seg = seg_img.get_fdata()
# seg[seg !=2] = 0
# seg[seg!=0] = 1
# print(seg.shape)
#
# bbox_lists = crop_seg(seg)
#
# for bbox in bbox_lists:
#
#     new_seg = seg[bbox[0]:bbox[1],bbox[2]:bbox[3],bbox[4]:bbox[5]]
#
#     print(new_seg.shape)
#
#     print(np.sum(new_seg)/np.sum(seg))






