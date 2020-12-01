######################################################
## Helper functions. Contains function to create bounding box. Padding
#####################################################
## Author: Assil Jwair
#####################################################

##################Imports#######################
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import ndimage
import numpy as np
from scipy.signal import argrelmax
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
import torchtuples as tt
import nibabel as nib
from pathlib import Path
import random
##################Imports#######################


def save_image(img, path_str):
    path = Path(path_str)
    path_parts = list(path.parts)
    for i, part in enumerate(path_parts):
        if part == 'preoperative_no_norm':
            path_parts[i] = 'preoperative_fcm'
    new_path = Path('/'.join(path_parts)[1:])
    Path(new_path.parent).mkdir(parents=True)
    nib.save(img, new_path)


def get_optimizer(name, net, lr_clinical=0.01, lr_mri=0.001):
    optim_dict = {
        'AdamW': tt.optim.AdamW(params=[{'params': net.seq.parameters(), 'lr': float(lr_clinical)}], lr=float(lr_mri)),
        'Adam': tt.optim.Adam(params=[{'params': net.seq.parameters(), 'lr': float(lr_clinical)}], lr=float(lr_mri)),
        'AdamWR': tt.optim.AdamWR(params=[{'params': net.seq.parameters(), 'lr': float(lr_clinical)}], lr=float(lr_mri))}
    return optim_dict[name]


def getPatientIdentifier(path):
    """
    Returns patient identifier of given path
    """
    return path.parts[2][9:]


def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()


def returnMontage(np_array, title):
    """
    Returns a montage of input 3D array

    :param np_array: 3D numpy array
    :param title: modality eg T1w
    :return: Montage
    """
    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 20))
    ax1.set(title=title)
    ax1.imshow(montage(np_array), cmap='gray')


def resize_image(img, spatial_size, order=1, mode="reflect", cval=0, clip=True, preserve_range=True, anti_aliasing=True):
    """
    Args:
        img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
    """
    resized = list()
    resized.append(
        resize(
            image=img,
            output_shape=spatial_size,
            order=order,
            mode=mode,
            cval=cval,
            clip=clip,
            preserve_range=preserve_range,
            anti_aliasing=anti_aliasing,
        )
    )
    return np.array(resized).astype(img.dtype)


def pad_image(image, correct_size):
    image_shape = image.shape
    for i, size in enumerate(image_shape):
        if size != correct_size:
            padding_size = correct_size - size
            if i == 0:
                image = np.pad(image, ((padding_size,0), (0,0), (0,0)), 'constant', constant_values=0)
            elif i == 1:
                image = np.pad(image, ((0,0), (padding_size,0), (0,0)), 'constant', constant_values=0)
            elif i == 2:
                image = np.pad(image, ((0,0), (0,0), (padding_size,0)), 'constant', constant_values=0)
    return image


def bbox_3d_new(mask, size):
    """
    Returns the bounding box of the tumor using ENT mask, bounding box size = (64,64,64)

    """
    
    center_of_mass_coordinates = ndimage.measurements.center_of_mass(mask)
    mask_shape = mask.shape
#     print(mask_shape)
    
    if center_of_mass_coordinates[0] is None:
        raise ValueError("Invalid mask")

    bbox = []
    coordinates_as_list = list(center_of_mass_coordinates)
    random_noise = random.choice(list(range(1,11)))
    random_direction = random.choice(list(range(3)))
    for i, coordinate in enumerate(coordinates_as_list):
        if i == random_direction:
            coordinates_as_list[i] += random_noise
    
    for i, coordinate in enumerate(tuple(coordinates_as_list)):
        min_coordinate = int(coordinate) - size//2
        max_coordinate = int(coordinate) + size//2
        if max_coordinate > mask_shape[i]:
            difference = max_coordinate - mask_shape[i]
            max_coordinate -= difference
            min_coordinate -= difference
            
        if min_coordinate < 0 or max_coordinate < 0:
            min_coordinate = 0
            # max_coordinate = size
        bbox.append(slice(min_coordinate, max_coordinate))

    return tuple(np.array(bbox))


def smooth_hist(data):
    """
    use KDE to get smooth estimate of histogram

    Args:
        data (np.ndarray): array of image data

    Returns:
        grid (np.ndarray): domain of the pdf
        pdf (np.ndarray): kernel density estimate of the pdf of data
    """
    data = data.flatten().astype(np.float64)
    bw = data.max() / 80

    kde = sm.nonparametric.KDEUnivariate(data)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    pdf = 100.0 * kde.density
    grid = kde.support

    return grid, pdf


def get_largest_mode(data):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data

    Returns:
        largest_peak (int): index of the largest peak
    """
    grid, pdf = smooth_hist(data)
    largest_peak = grid[np.argmax(pdf)]
    return largest_peak


def get_last_mode(data, rare_prop=96, remove_tail=True):
    """
    gets the last (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data
        rare_prop (float): if remove_tail, use the proportion of hist above
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        last_peak (int): index of the last peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, rare_prop)
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid, pdf = smooth_hist(data)
    maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    last_peak = grid[maxima[-1]]
    return last_peak


def get_first_mode(data, rare_prop=99, remove_tail=True):
    """
    gets the first (reliable) peak in the histogram

    Args:
        data (np.ndarray): image data
        rare_prop (float): if remove_tail, use the proportion of hist above
        remove_tail (bool): remove rare portions of histogram
            (included to replicate the default behavior in the R version)

    Returns:
        first_peak (int): index of the first peak
    """
    if remove_tail:
        rare_thresh = np.percentile(data, rare_prop)
        which_rare = data >= rare_thresh
        data = data[which_rare != 1]
    grid, pdf = smooth_hist(data)
    maxima = argrelmax(pdf)[0]  # for some reason argrelmax returns a tuple, so [0] extracts value
    first_peak = grid[maxima[0]]
    return first_peak


def gmm_class_mask(img, brain_mask=None, contrast='t1', return_wm_peak=True, hard_seg=False):
    """
    get a tissue class mask using gmms (or just the WM peak, for legacy use)
    Args:
        img (nibabel.nifti1.Nifti1Image): target img
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
            (none if already skull-stripped)
        contrast (str): string to describe img's MR contrast
        return_wm_peak (bool): if true, return only the wm peak
        hard_seg (bool): if true and return_wm_peak false, then return
            hard segmentation of tissue classes
    Returns:
        if return_wm_peak true:
            wm_peak (float): represents the mean intensity for WM
        else:
            mask (np.ndarray):
                if hard_seg, then mask is the same size as img
                else, mask is the same size as img * 3, where
                the new dimensions hold the probabilities of tissue class
    """
    img_data = img.get_fdata()
    if brain_mask is not None:
        mask_data = brain_mask.get_fdata() > 0
    else:
        mask_data = img_data > img_data.mean()

    brain = np.expand_dims(img_data[mask_data].flatten(), 1)
    gmm = GaussianMixture(3)
    gmm.fit(brain)

    if return_wm_peak:
        means = sorted(gmm.means_.T.squeeze())
        if contrast.lower() == 't1':
            wm_peak = means[2]
        elif contrast.lower() == 'flair':
            wm_peak = means[1]
        elif contrast.lower() == 't2':
            wm_peak = means[0]
        return wm_peak
    else:
        classes_ = np.argsort(gmm.means_.T.squeeze())
        if contrast.lower() == 't1':
            classes = [classes_[0], classes_[1], classes_[2]]
        elif contrast.lower() == 'flair':
            classes = [classes_[0], classes_[2], classes_[1]]
        elif contrast.lower() == 't2':
            classes = [classes_[2], classes_[1], classes_[0]]
        if hard_seg:
            tmp_predicted = gmm.predict(brain)
            predicted = np.zeros(tmp_predicted.shape)
            for i, c in enumerate(classes):
                predicted[tmp_predicted == c] = i + 1
            mask = np.zeros(img_data.shape)
            mask[mask_data] = predicted + 1
        else:
            predicted_proba = gmm.predict_proba(brain)
            mask = np.zeros((*img_data.shape, 3))
            for i, c in enumerate(classes):
                mask[mask_data, i] = predicted_proba[:, c]
        return mask