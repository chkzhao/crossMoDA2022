from skimage.measure import label, regionprops
import numpy as np
# import matplotlib.pyplot as plt
# import os

# def saveTransformedImages(img):
#     save_path = "../transformed_images"
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)

#     plt.figure("prob", (6, 6))
#     plt.title("label")
#     plt.imshow(img[:, :, 37], interpolation="none")
#     plt.savefig(os.path.join(save_path, "problem.png"))

def crop_seg(seg, coc_size = [32, 32, 16], center=False):
    W,H,D = seg.shape
    lbl_0 = label(seg)
    props = regionprops(lbl_0)
    bbox_lists = []
    if len(props) == 0:
        x0 = np.random.choice(W-coc_size[0],1).item()
        x1 = x0 + coc_size[0]

        y0 = np.random.choice(H - coc_size[1], 1).item()
        y1 = y0 + coc_size[1]

        z0 = np.random.choice(D - coc_size[2], 1).item()
        z1 = z0 + coc_size[2]

        x_b = max(min(np.random.choice(coc_size[0] + 1 - (x1 - x0), 1).item(), x0), x0 + coc_size[0] - W)
        y_b = max(min(np.random.choice(coc_size[1] + 1 - (y1 - y0), 1).item(), y0), y0 + coc_size[1] - H)
        z_b = max(min(np.random.choice(coc_size[2] + 1 - (z1 - z0), 1).item(), z0), z0 + coc_size[2] - D)

        x0_b = x0 - x_b
        x1_b = x1 + coc_size[0] - (x1 - x0) - x_b

        y0_b = y0 - y_b
        y1_b = y1 + coc_size[1] - (y1 - y0) - y_b

        z0_b = z0 - z_b
        z1_b = z1 + coc_size[2] - (z1 - z0) - z_b

        bbox_lists.append([x0_b, x1_b, y0_b, y1_b, z0_b, z1_b])
    else:
        # print("The length of props is {}".format(len(props)))
        for prop in props:
            bbox = prop.bbox

            x0, y0, z0, x1, y1, z1 = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]

            # print("{}\t{}\t{}\t{}\t{}\t{}".format(x0, y0, z0, x1, y1, z1))
            # if (coc_size[2] + 1 - (z1 - z0)) <= 0:
            #     print("Saving the problem label...")
            #     saveTransformedImages(seg)

            if (coc_size[0] + 1 - (x1 - x0)) <= 0:
                x_b = 0
            else:
                x_b = max(min(np.random.choice(coc_size[0] + 1 - (x1 - x0), 1).item() if not center else int((coc_size[0] + 1 - (x1 - x0))/2.0) , x0), x0 + coc_size[0] - W)
            if (coc_size[1] + 1 - (y1 - y0)) <= 0:
                y_b = 0
            else:
                y_b = max(min(np.random.choice(coc_size[1] + 1 - (y1 - y0), 1).item() if not center else int((coc_size[1] + 1 - (y1 - y0))/2.0), y0), y0 + coc_size[1] - H)
            if (coc_size[2] + 1 - (z1 - z0)) <= 0:
                z_b = 0
            else:
                z_b = max(min(np.random.choice(coc_size[2] + 1 - (z1 - z0), 1).item() if not center else int((coc_size[2] + 1 - (z1 - z0))/2.0), z0), z0 + coc_size[2] - D)

            x0_b = x0 - x_b
            x1_b = x1 + coc_size[0] - (x1 - x0) - x_b

            y0_b = y0 - y_b
            y1_b = y1 + coc_size[1] - (y1 - y0) - y_b

            z0_b = z0 - z_b
            z1_b = z1 + coc_size[2] - (z1 - z0) - z_b

            bbox_lists.append([x0_b, x1_b, y0_b, y1_b, z0_b, z1_b])

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






