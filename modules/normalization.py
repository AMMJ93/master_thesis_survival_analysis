#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
intensity_normalization.normalize.zscore

normalize an image by simply subtracting the mean
and dividing by the standard deviation of the whole brain

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 30, 2018
"""

from __future__ import print_function, division

import logging

import nibabel as nib
import torch

from skfuzzy import cmeans
from modules.utils import *
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    from sklearn.mixture import GMM as GaussianMixture


def zscore_normalize(img, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain imag
        e
        mask (nibabel.nifti1.Nifti1Image): brain mask for img0

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_fdata()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_fdata()
    elif mask == 'nomask':
        mask_data = img_data == img_data
    else:
        mask_data = img_data > img_data.mean()
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    return normalized

def zscore_normalize2(img, mask=None):
    """
    normalize a target image by subtracting the mean of the whole brain
    and dividing by the standard deviation

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain imag
        e
        mask (nibabel.nifti1.Nifti1Image): brain mask for img0

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
    """

    img_data = img.get_fdata()
    if mask is not None and not isinstance(mask, str):
        mask_data = mask.get_fdata()
    elif mask == 'nomask':
        mask_data = img_data > 0
    logical_mask = mask_data == 1  # force the mask to be logical type
    mean = img_data[logical_mask].mean()
    std = img_data[logical_mask].std()
    zero_values = 0 - mean / std
    normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
    normalized = normalized.get_fdata()
    zero_indices = normalized == zero_values
    normalized[zero_indices] = 0
    return normalized


def fcm_class_mask(img, brain_mask=None, hard_seg=False):
    """
    creates a mask of tissue classes for a target brain with fuzzy c-means
    Args:
        img (nibabel.nifti1.Nifti1Image): target image (must be T1w)
        brain_mask (nibabel.nifti1.Nifti1Image): mask covering the brain of img
            (none if already skull-stripped)
        hard_seg (bool): pick the maximum membership as the true class in output
    Returns:
        mask (np.ndarray): membership values for each of three classes in the image
            (or class determinations w/ hard_seg)
    """
    img_data = img.get_fdata()
    if brain_mask is not None:
        mask_data = brain_mask.get_fdata() > 0
    else:
        mask_data = img_data > img_data.mean()
    [t1_cntr, t1_mem, _, _, _, _, _] = cmeans(img_data[mask_data].reshape(-1, len(mask_data[mask_data])),
                                              3, 2, 0.005, 50)
    t1_mem_list = [t1_mem[i] for i, _ in sorted(enumerate(t1_cntr), key=lambda x: x[1])]  # CSF/GM/WM
    mask = np.zeros(img_data.shape + (3,))
    for i in range(3):
        mask[..., i][mask_data] = t1_mem_list[i]
    if hard_seg:
        tmp_mask = np.zeros(img_data.shape)
        tmp_mask[mask_data] = np.argmax(mask[mask_data], axis=1) + 1
        mask = tmp_mask
    return mask

def fcm_normalize(img, tissue_mask, norm_value=1):
    """
    Use FCM generated mask to normalize some specified tissue of a target image

    Args:
        img (nibabel.nifti1.Nifti1Image): target MR brain image
        tissue_mask (nibabel.nifti1.Nifti1Image): tissue mask for img
        norm_value (float): value at which to place the tissue mean

    Returns:
        normalized (nibabel.nifti1.Nifti1Image): img with specified tissue mean at norm_value
    """

    img_data = img.get_fdata()
    tissue_mask_data = tissue_mask
    tissue_mean = img_data[tissue_mask_data > 0].mean()
    normalized = nib.Nifti1Image((img_data / tissue_mean) * norm_value,
                                 img.affine, img.header)
    return normalized


def find_tissue_mask(img, brain_mask, threshold=0.8, tissue_type='wm'):
    """
    find tissue mask using FCM with a membership threshold

    Args:
        img (nibabel.nifti1.Nifti1Image): target img
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        threshold (float): membership threshold
        tissue_type (str): find the mask of this tissue type (wm, gm, or csf)

    Returns:
        tissue_mask_nifti (nibabel.nifti1.Nifti1Image): tissue mask for img
    """
    tissue_to_int = {'csf': 0, 'gm': 1, 'wm': 2}
    t1_mem = fcm_class_mask(img, brain_mask)
    tissue_mask = t1_mem[..., tissue_to_int[tissue_type]] > threshold
#     tissue_mask_nifti = nib.Nifti1Image(tissue_mask, img.affine, img.header)
    return torch.from_numpy(tissue_mask)


def whitestripe(img, contrast, mask=None, width=0.05, width_l=None, width_u=None):
    """
    find the "(normal appearing) white (matter) stripe" of the input MR image
    and return the indices
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        contrast (str): contrast of img (e.g., T1)
        mask (nibabel.nifti1.Nifti1Image): brainmask for img (None is default, for skull-stripped img)
        width (float): width quantile for the "white (matter) stripe"
        width_l (float): lower bound for width (default None, derives from width)
        width_u (float): upper bound for width (default None, derives from width)
    Returns:
        ws_ind (np.ndarray): the white stripe indices (boolean mask)
    """
    if width_l is None and width_u is None:
        width_l = width
        width_u = width
    img_data = img.get_fdata()
    if mask is not None:
        mask_data = mask.get_fdata()
        masked = img_data * mask_data
        voi = img_data[mask_data == 1]
    else:
        masked = img_data
        voi = img_data[img_data > img_data.mean()]
    if contrast.lower() in ['t1', 'last']:
        mode = get_last_mode(voi)
    elif contrast.lower() in ['t2', 'flair', 'largest']:
        mode = get_largest_mode(voi)
    elif contrast.lower() in ['md', 'first']:
        mode = get_first_mode(voi)
    img_mode_q = np.mean(voi < mode)
    ws = np.percentile(voi, (max(img_mode_q - width_l, 0) * 100, min(img_mode_q + width_u, 1) * 100))
    ws_ind = np.logical_and(masked > ws[0], masked < ws[1])
    return ws_ind


def whitestripe_norm(img, indices):
    """
    use the whitestripe indices to standardize the data (i.e., subtract the
    mean of the values in the indices and divide by the std of those values)
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        indices (np.ndarray): whitestripe indices (see whitestripe func)
    Returns:
        norm_img (nibabel.nifti1.Nifti1Image): normalized image in nifti format
    """
    img_data = img.get_fdata()
    mu = np.mean(img_data[indices])
    sig = np.std(img_data[indices])
    norm_img_data = (img_data - mu)/sig
    norm_img = nib.Nifti1Image(norm_img_data, img.affine, img.header)
    return norm_img


def gmm_normalize(img, brain_mask=None, norm_value=1, contrast='t1', bg_mask=None, wm_peak=None):
    """
    normalize the white matter of an image using a GMM to find the tissue classes
    Args:
        img (nibabel.nifti1.Nifti1Image): target MR image
        brain_mask (nibabel.nifti1.Nifti1Image): brain mask for img
        norm_value (float): value at which to place the WM mean
        contrast (str): MR contrast type for img
        bg_mask (nibabel.nifti1.Nifti1Image): if provided, use to zero bkgd
        wm_peak (float): previously calculated WM peak
    Returns:
        normalized (nibabel.nifti1.Nifti1Image): gmm wm peak normalized image
    """

    if wm_peak is None:
        wm_peak = gmm_class_mask(img, brain_mask=brain_mask, contrast=contrast)

    img_data = img.get_fdata()
    norm_data = (img_data / wm_peak) * norm_value
    norm_data[norm_data < 0.1] = 0.0

    if bg_mask is not None:
        masked_image = norm_data * bg_mask.get_fdata()
    else:
        masked_image = norm_data

    normalized = nib.Nifti1Image(masked_image, img.affine, img.header)
    return normalized


def main(img, contrast='t1', method='zscore', fcm=None):
    """
    Written by Assil Jwair
    :param img: Input MRI sequence
    :param contrast: What sequence?
    :param method: What method?
    :param fcm: FCM yes/no
    :return: normalized image (Nifti)
    """
    contrast_mapping = {'T1c': 't1', 'T1w': 't1', 'FLR': 'flair', 'T2w': 't2'}
    if method == 'zscore':
        return zscore_normalize(img, mask='nomask')
    elif method == 'fcm':
        if contrast == 'T1c':
            return find_tissue_mask(img, None)
        else:
            return fcm_normalize(img, fcm)
    elif method == 'whitestripe':
        mapped_contrast = contrast_mapping[contrast]
        wd_ind = whitestripe(img, mapped_contrast, mask=None, width=0.05, width_l=None, width_u=None)
        return whitestripe_norm(img, wd_ind)
    elif method == 'gmm':
        mapped_contrast = contrast_mapping[contrast]
        return gmm_normalize(img, brain_mask=None, norm_value=1, contrast= mapped_contrast, bg_mask=None, wm_peak=None)
