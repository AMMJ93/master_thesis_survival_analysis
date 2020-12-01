######################################################
## Creates a PyTorch Dataset. Handles splits, bounding 
## box,transformations, data augmentation, transformations,
## montage plotting and more. Requires 4 modalities. 
#####################################################
## Author: Assil Jwair
## Version: 8.0
#####################################################

##################Imports#######################
import sys
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import nibabel as nib
from modules.utils import returnMontage
from modules.utils import bbox_3d_new, pad_image, save_image
import numpy as np
import torchtuples as tt
import random
from modules.normalization import zscore_normalize as zscore_norm
from modules.normalization import fcm_normalize as fcm_norm
from modules.normalization import find_tissue_mask as fcm_find
from modules.normalization import main as norm_main


##################Imports#######################

# def collate_fn(batch):
#     """Stacks the entries of a nested tuple"""
#     return tt.tuplefy(batch).stack()

class Dataset(Dataset):
    """
    Generic dataset to handle dictionary format data, it can operate transforms for specific fields.

    """

    def __init__(self, data, normalize=False, montage=False, phase=None, only_modality=False, size=64, save=False,
                 save_fcm=False):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            montage (Boolean): returns montage plot if True
            phase (String): defines dataset type (train, test, validation) to generate
            only_modality (String): returns only selected phase (T1c, T1w, FLR, ENT)
            size (Int): Size of bounding box, default 64
            save (Boolean): Saves bounding boxes as 3D nifti image if True
            normalize (String): What normalization technique to use (Zscore, Whitestripe.. )
        """
        self.data = data
        self.normalize = normalize
        self.montage = montage
        self.phase = phase
        self.only_modality = only_modality
        self.size = size
        self.save = save
        self.save_fcm = save_fcm

        if phase == 'train':
            self.training = self.data[:int(len(data) * 0.7)]
            sys.stdout.write('Training Size: {}\n'.format(len(self.training)))

        elif phase == 'test':
            self.testing = self.data[int(len(data) * 0.8):]
            sys.stdout.write('Test Size: {}\n'.format(len(self.testing)))

        elif phase == 'val':
            self.validation = self.data[int(len(data) * 0.7):int(len(data) * 0.8)]
            sys.stdout.write('Validation Size: {}\n'.format(len(self.validation)))

        elif phase == 'all':
            self.all_data = self.data
            sys.stdout.write('All data Size: {}\n'.format(len(self.all_data)))

    def __len__(self):
        if self.phase == 'train':
            return len(self.training)
        elif self.phase == 'test':
            return len(self.testing)
        elif self.phase == 'val':
            return len(self.validation)
        elif self.phase == 'all':
            return len(self.all_data)

    def __getitem__(self, index):
        if self.phase == 'train':
            self.tissue_mask = None
            data = self.training[index]
            random_draw_rotate = random.choices(population=[0, 1, 2], weights=[0.4, 0.3, 0.3], k=1)[
                0]  # If 1: Flip image lr, if 2: Flip image up
            dropped = False

        elif self.phase == 'test':
            data = self.testing[index]
            random_draw_rotate = False

        elif self.phase == 'val':
            data = self.validation[index]
            random_draw_rotate = False

        elif self.phase == 'all':
            data = self.all_data[index]
            random_draw_rotate = False

        modalities = ['T1c', 'T1w', 'FLR', 'ENT', 'T2w']
        return_obj = []
        bbox = None
        missing_modalities = []
        for modality in modalities:
            if modality == 'ENT' and data[modality] is not None:
                ent_map = nib.load(data[modality]).get_fdata()  # Load ENT map
                try:
                    bbox = bbox_3d_new(ent_map, self.size)  # Calculate bound box coordinates
                except:
                    raise ValueError("Not able to generate bounding box, check mask dimensions")
        for modality in modalities:
            if modality != 'ENT':
                path = data[modality]  # Get path of sequence
                if path is not None:   # Check whether sequence exists for patients
                    if self.phase == 'train':  # Only drop for train patients
                        random_draw_drop = random.choices(population=[0, 1], weights=[0.8, 0.2], k=1)[0]
                    else:
                        random_draw_drop = False
                    if self.normalize:  # If normalizing
                        nifti_img = nib.load(data[modality])
                        min_ = nifti_img.get_fdata().min()
                        max_ = nifti_img.get_fdata().max()
                        if min_ == -0 and max_ == -0:  # Check if sequence is "present" but invalid
                            image = nib.load(data[modality])
                        else:
                            if self.normalize == 'fcm' and modality == 'T1c':  # FCM needs special approach..
                                self.tissue_mask = norm_main(nifti_img, contrast=modality, method=self.normalize)
                                image_nib = norm_main(nifti_img, contrast=modality, method='zscore')
                                image = image_nib.get_fdata()

                            elif self.normalize == 'fcm' and modality != 'T1c':
                                image_nib = norm_main(nifti_img, contrast=modality, method=self.normalize,
                                                      fcm=self.tissue_mask)
                                image = image_nib.get_fdata()

                            else:
                                image = norm_main(nifti_img, contrast=modality, method=self.normalize).get_fdata()

                            if self.save_fcm:
                                save_image(image_nib, path)

                    elif not self.normalize:  # If normalization is not needed
                        image = nib.load(data[modality]).get_fdata()
                        min_ = image.min()
                        max_ = image.max()

                    if min_ == -0 and max_ == -0:  # If image is existing but invalid
                        missing_modalities.append(modality)
                        zeros = (self.size, self.size, self.size)
                        image_numpy = np.zeros(zeros)
                        return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))

                    else:
                        if bbox is not None:
                            image_numpy = np.array(image[bbox])  # Get bounding box out of 3D MRI
                            if image_numpy.shape != (self.size, self.size, self.size):  # If not correct size, pad image
                                image_numpy = pad_image(image_numpy, self.size)
                        else:
                            image_numpy = np.array(image)
                        # Handles random dropping of sequence
                        if random_draw_drop == 1 and modality != 'T1c' and not dropped:
                            zeros = (self.size, self.size, self.size)
                            image_numpy = np.zeros(zeros)
                            return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))
                            dropped = True
                        # Handles random rotating
                        elif random_draw_rotate == 1:
                            image_numpy = np.fliplr(image_numpy)
                            return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))
                        elif random_draw_rotate == 2:
                            image_numpy = np.flipud(image_numpy)
                            return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))
                        else:
                            return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))

                    if self.montage:  # Returns 2D slices to visualize patients sequences :)
                        returnMontage(image_numpy, modality)
                    if self.save:  # Saves generated bounding boxes if wanted
                        ni_img = nib.Nifti1Image(image_numpy, np.eye(4))
                        nib.save(ni_img, '{}_bounding_box_{}.nii.gz'.format(data['id'], modality))

                else:  # If sequence is missing, make a box of zeros
                    missing_modalities.append(modality)
                    zeros = (self.size, self.size, self.size)
                    image_numpy = np.zeros(zeros)
                    return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))

        if not self.montage:
            time, event = tuple((torch.tensor(np.array(data['group'])), torch.tensor(np.array(data['event']))))
            id_ = data['id']

            if self.only_modality:
                torch_images = (return_obj[0][None])
                if torch_images.size() != torch.Size([1, self.size, self.size, self.size]):  # Check if bounding box size is correct
                    return id_

            else:
                torch_images = torch.cat(
                    tuple((return_obj[0][None], return_obj[1][None], return_obj[2][None], return_obj[3][None])))  # Stack images
                if torch_images.size() != torch.Size([4, self.size, self.size, self.size]):  # Check that patient has 4 sequences
                    print(torch_images.size())
                    return id_

            return tuple((torch_images.float(), (time, event.float())))
