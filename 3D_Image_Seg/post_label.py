import pdb
import cv2
import os
import numpy as np
import nibabel as nib
import sys
import time
import logging
import logging.handlers
import pydensecrf.densecrf as dcrf

def get_crf_img(inputs, outputs):
    for i in range(outputs.shape[0]):
        img = inputs[i]
        softmax_prob = outputs[i]
        unary = unary_from_softmax(softmax_prob)
        unary = np.ascontiguousarray(unary)
        d = dcrf.DenseCRF(img.shape[0] * img.shape[1], 2)
        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=(10,10), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(sdims=(50,50), schan=(20,20,20),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
        if i == 0:
            crf = np.expand_dims(res,axis=0)
        else:
            res = np.expand_dims(res,axis=0)
            crf = np.concatenate((crf,res),axis=0)
    return crf


def erode_dilate(outputs, kernel_size=7):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    outputs = outputs.astype(np.uint8)
    for i in range(outputs.shape[0]):
        img = outputs[i]
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        outputs[i] = img
    return outputs


def post_process(args, inputs, outputs, input_path=None,
                 crf_flag=True, erode_dilate_flag=True,
                 save=True, overlap=True):
    inputs = (np.array(inputs.squeeze()).astype(np.float32)) * 255
    inputs = np.expand_dims(inputs, axis=3)
    inputs = np.concatenate((inputs,inputs,inputs), axis=3)
    outputs = np.array(outputs)

    # Conditional Random Field
    if crf_flag:
        outputs = get_crf_img(inputs, outputs)
    else:
        outputs = outputs.argmax(1)

    # Erosion and Dilation
    if erode_dilate_flag:
        outputs = erode_dilate(outputs, kernel_size=7)
    if save == False:
        return outputs

    outputs = outputs*255
    for i in range(outputs.shape[0]):
        path = input_path[i].split('/')
        output_folder = os.path.join(args.output_root, path[-2])
        try:
            os.mkdir(output_folder)
        except:
            pass
        output_path = os.path.join(output_folder, path[-1])
        if overlap:
            img = outputs[i]
            img = np.expand_dims(img, axis=2)
            zeros = np.zeros(img.shape)
            img = np.concatenate((zeros,zeros,img), axis=2)
            img = np.array(img).astype(np.float32)
            img = inputs[i] + img
            if img.max() > 0:
                img = (img/img.max())*255
            else:
                img = (img/1) * 255
            cv2.imwrite(output_path, img)
        else:
            img = outputs[i]
            cv2.imwrite(output_path, img)
    return None


from scipy import ndimage
import nibabel as nib
import os
import numpy as np

def get_largest_two_component(img, print_info = False, threshold = None):
    """
    Get the largest two components of a binary volume
    inputs:
        img: the input 3D volume
        threshold: a size threshold
    outputs:
        out_img: the output volume
    """
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    if(print_info):
        print('component size', sizes_list)
    if(len(sizes) == 1):
        out_img = img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            if(max_size2*4 > max_size1):
                component1 = (component1 + component2) > 0
            out_img = component1
    return out_img

def fill_holes(img):
    """
    filling small holes of a binary volume with morphological operations
    """
    neg = 1 - img
    s = ndimage.generate_binary_structure(3, 1)  # iterate structure
    labeled_array, numpatches = ndimage.label(neg, s)  # labeling
    sizes = ndimage.sum(neg, labeled_array, range(1, numpatches + 1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    max_size = sizes_list[-1]
    max_label = np.where(sizes == max_size)[0] + 1
    component = labeled_array == max_label
    return 1 - component

struct = ndimage.generate_binary_structure(3, 2)

image_path = './label'
save_path = './post_label'
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i in range(211,243):
    print(i)
    label_nib = nib.load(
        os.path.join(image_path, 'crossmoda_' + str(i) + '_Label.nii.gz'))
    label = label_nib.get_fdata()

    label_1 = label * 1
    label_1[label_1 != 1] = 0

    label_2 = label * 1
    label_2[label_2 != 2] = 0

    label_1 = ndimage.morphology.binary_closing(label_1, structure=struct)
    label_1 = get_largest_two_component(label_1).astype(np.int16)

    label_2 = ndimage.morphology.binary_closing(label_2, structure=struct).astype(np.int16)
    # label_2 = get_largest_two_component(label_2)

    new_label = label_1 + label_2 * 2

    new_label = nib.Nifti1Image(new_label, label_nib.affine)
    new_label = new_label.__class__(new_label.dataobj[:], label_nib.affine, label_nib.header)

    nib.save(new_label, os.path.join(save_path, 'crossmoda_' + str(i) + '_Label.nii.gz'))
