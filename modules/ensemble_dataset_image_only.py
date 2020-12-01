import sys
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
import nibabel as nib
from modules.utils import returnMontage
from modules.utils import bbox_3d_new, pad_image
import numpy as np
import torchtuples as tt
import random
from modules.normalization import zscore_normalize as zscore_norm
from modules.normalization import fcm_normalize as fcm_norm
from modules.normalization import find_tissue_mask as fcm_find
from modules.normalization import main as norm_main


# def collate_fn(batch):
#     """Stacks the entries of a nested tuple"""
#     return tt.tuplefy(batch).stack()

class Dataset(Dataset):
    """
    Generic dataset to handle dictionary format data, it can operate transforms for specific fields.

    """

    def __init__(self, data, clinical=False, normalize=False, montage=False, phase=None, only_modality=False, targets=False, size=64, save=False):
        """
        Args:
            data (Iterable): input data to load and transform to generate dataset for model.
            montage (Boolean): returns montage plot if True
            phase (String): defines dataset type (train, test, validation) to generate
            only_modality (String): returns only selected phase (T1c, T1w, FLR, ENT)
            size (Int): Size of bounding box, default 64
            save (Boolean): Saves bounding boxes as 3D nifti image if True

        """
        self.data = data
        self.clinical = clinical
        self.normalize = normalize
        self.montage = montage
        self.phase = phase
        self.only_modality = only_modality
        self.targets = targets
        self.size = size
        self.save = save

        if phase == 'test':
            self.testing = self.data[int(len(data) * 0.8):]
            self.clinical = self.clinical[int(len(clinical) * 0.8):]
            sys.stdout.write('Test Size: {}\n'.format(len(self.testing)))

    def __len__(self):
        return len(self.testing)

    def __getitem__(self, index):

        data = self.testing[index]
        clinical_data = self.clinical[index]

        modalities = ['T1c', 'T1w','FLR', 'ENT', 'T2w']
        return_obj = []
        bbox = None
        missing_modalities = []
        if not self.targets:
            for modality in modalities:
                if modality == 'ENT' and data[modality] is not None:
                    ent_map = nib.load(data[modality]).get_fdata()
                    try:
                        bbox = bbox_3d_new(ent_map, self.size)
                    except:
                        raise ValueError("Not able to generate bounding box, check mask dimensions")
            for modality in modalities:
                if modality != 'ENT':
                    path = data[modality]
                    if path is not None:
                        if self.normalize:
                            nifti_img = nib.load(data[modality])
                            min_ = nifti_img.get_fdata().min()
                            max_ = nifti_img.get_fdata().max()
                            if min_ == -0 and max_ == -0:
                                image = nib.load(data[modality])
                            else:
                                if self.normalize == 'fcm' and modality == 'T1c':
                                    self.tissue_mask = norm_main(nifti_img, contrast=modality, method=self.normalize)
                                    image = norm_main(nifti_img, contrast=modality, method='zscore').get_fdata()
                                elif self.normalize == 'fcm' and modality != 'T1c':
                                    image = norm_main(nifti_img, contrast=modality, method=self.normalize, fcm=self.tissue_mask).get_fdata()
                                else:
                                    image = norm_main(nifti_img, contrast=modality, method=self.normalize).get_fdata()

                        elif not self.normalize:
                            image = nib.load(data[modality]).get_fdata()
                            min_ = image.min()
                            max_ = image.max()

                        if min_ == -0 and max_ == -0:
                            missing_modalities.append(modality)
                            zeros = (self.size, self.size, self.size)
                            image_numpy = np.zeros(zeros)
                            return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))

                        else:
                            if bbox is not None:
                                image_numpy = np.array(image[bbox])
                                if image_numpy.shape != (64, 64, 64):
                                    image_numpy = pad_image(image_numpy, self.size)
                            else:
                                image_numpy = np.array(image)

                            return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))

                        if self.montage:
                            returnMontage(image_numpy, modality)
                        if self.save:
                            ni_img = nib.Nifti1Image(image_numpy, np.eye(4))
                            nib.save(ni_img, '{}_bounding_box_{}.nii.gz'.format(data['id'], modality))

                    else:
                        missing_modalities.append(modality)
                        zeros = (self.size, self.size, self.size)
                        image_numpy = np.zeros(zeros)
                        return_obj.append(torch.as_tensor(np.ascontiguousarray(image_numpy)))

        if not self.montage:

            if self.targets:
                return np.array(([x['group'] for x in self.testing], [x['event'] for x in self.testing], [x['surv'] for x in self.testing]))
            
            else:
                time, event = tuple((torch.tensor(np.array(data['group'])), torch.tensor(np.array(data['event']))))
                id_ = data['id']

                if self.only_modality:
                    torch_images = (return_obj[0][None])
                    if torch_images.size() != torch.Size([1, self.size, self.size, self.size]):
                        return id_

                else:
                    torch_images = torch.cat(
                        tuple((return_obj[0][None], return_obj[1][None], return_obj[2][None], return_obj[3][None])))
                    if torch_images.size() != torch.Size([4, self.size, self.size, self.size]):
                        print(torch_images.size())
                        return id_                

                return tuple((torch_images.float(), torch.tensor(clinical_data).float()))




