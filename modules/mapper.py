######################################################
## Creates dictionary per subject with mapping
## of sequence to file location. Also survival time,
## event. The input for PyTorch dataset class!
#####################################################
## Author: Assil Jwair
## Version: 4.0
#####################################################

import numpy as np
from pathlib import Path
from modules.utils import getPatientIdentifier
import json
import sys


class FileMapper:
    def __init__(self, mri_paths, id_mapping_json, normalized=False, fcm=False, brats_path=False, postop_path=False):
        """
        Args:
            mri_paths (String): Path to MRI dataset.
            id_mapping_json (String): Path to id -> survival group json file
            normalized (Boolean): Roelants already normalized files?
            fcm (Boolean): If FCM normalized.
            brats_path (String): Path to Brats dataset (same folder structure as mri_paths).
        """
        self.mri_paths = mri_paths
        self.mapping = []
        self.normalized = normalized
        self.fcm = fcm
        self.brats_path = brats_path
        self.postop_path = postop_path

        with open(id_mapping_json) as json_file:
            data = json.load(json_file)
            self.id_mapping = json.loads(data)

    def get_subject_id(self, path):
        """
        Returns patient identifier of given path
        """
        if path.parts[2][0:8] != 'BMIAXNAT':
            if path.parts[3][0:8] == 'BMIAXNAT':
                    return path.parts[3]
            else:
                raise ValueError('Invalid path')
        else:
            return path.parts[2]

    def get_subject_id_brats(self, path):
        """
        Returns patient identifier of given path for BraTS dataset
        """
        return path.parts[2]

    def generate_mapping(self):
        """
        Generates mapping for preoperative images.
        :return: list of dictionaries
        """
        for path in self.mri_paths:
            subject_id = self.get_subject_id(path)
            if subject_id in list(self.id_mapping['surv'].keys()):
                surv = self.id_mapping['surv'][subject_id]
                event = self.id_mapping['DeathObserved'][subject_id]
                group = self.id_mapping['group'][subject_id]
                subject_dict = {'id': subject_id}
                sub_folders = [x for x in path.iterdir() if x.is_dir()]
                modalities = ['T1w', 'T1c', 'ENT', 'FLR', 'T2w']
                available_modalities = []
                for folder in sub_folders:
                    if self.normalized:
                        image = list(folder.rglob('*.nii.gz'))[0].absolute()
                    else:
                        image = list(folder.rglob('*masked.nii.gz'))[0].absolute()
                    if self.fcm:
                        intensity = folder.parts[4]
                    else:
                        intensity = folder.parts[3]
                    if intensity in modalities:
                        available_modalities.append(intensity)
                        subject_dict[intensity] = str(image)

                for modality in modalities:
                    if modality not in available_modalities:
                        subject_dict[modality] = None

                subject_dict['surv'] = surv
                subject_dict['event'] = event
                subject_dict['group'] = group
                self.mapping.append(subject_dict)
        sys.stdout.write('Number of patients/folders: {}\n'.format(len(self.mapping)))
        if not self.postop_path:
            return self.mapping

    def add_postop(self):
        """
        Adds the postop paths to self.mapping. First generate (preop) mapping!
        """
        count = 0
        all_modalities = ['T1w', 'T1c', 'ENT', 'FLR', 'T2w', 'postop_T1w', 'postop_T1c', 'postop_FLR', 'postop_T2w']
        for path in self.postop_path:
            subject_id = self.get_subject_id(path)
            if subject_id in list(self.id_mapping['surv'].keys()):
                for i, subj in enumerate(self.mapping):
                    id_ = subj['id']
                    if id_ == subject_id:
                        count += 1
                        sub_folders = [x for x in path.iterdir() if x.is_dir()]
                        modalities = ['postop_T1w', 'postop_T1c', 'postop_ENT', 'postop_FLR', 'postop_T2w']
                        available_modalities = []
                        for folder in sub_folders:
                            if self.normalized:
                                image = list(folder.rglob('*.nii.gz'))[0].absolute()
                            else:
                                image = list(folder.rglob('*masked.nii.gz'))[0].absolute()
                            if self.fcm:
                                intensity = folder.parts[4]
                            else:
                                intensity = folder.parts[3]
                            if intensity in modalities:
                                available_modalities.append(intensity)
                                self.mapping[i][intensity] = str(image)
                        for modality in modalities:
                            if modality not in available_modalities:
                                self.mapping[i][intensity] = None

        for i, subj in enumerate(self.mapping):
            keys = list(subj.keys())
            for modality in all_modalities:
                if modality not in keys:
                    self.mapping[i][modality] = None
        sys.stdout.write('Number of patients with postop added: {}\n'.format(count))
        return self.mapping

    def generate_mapping_brats(self):
        """
        Generates mapping for brats dataset
        :return: list of dictionaries
        """
        brats_mapping = []
        picture_mapping = self.generate_mapping()
        for path in self.brats_path:
            subject_id = self.get_subject_id_brats(path)
            if subject_id in list(self.id_mapping['surv'].keys()):
                surv = self.id_mapping['surv'][subject_id]
                event = self.id_mapping['DeathObserved'][subject_id]
                group = self.id_mapping['group'][subject_id]
                subject_dict = {'id': subject_id}
                sub_files = [x for x in path.iterdir() if not x.is_dir()]
                modalities = ['t1', 't2', 't1ce', 'flair', 'seg']
                mapper = {'t1': 'T1w', 't1ce': 'T1c', 't2': 'T2w', 'flair': 'FLR', 'seg': 'ENT'}
                available_modalities = []
                for file in sub_files:
                    for modality in modalities:
                        if modality == 't1':
                            modality = 't1.nii.gz'
                        if modality in file.parts[3]:
                            if modality == 't1.nii.gz':
                                mapped_modality = mapper['t1']
                            else:
                                mapped_modality = mapper[modality]
                            available_modalities.append(mapped_modality)
                            subject_dict[mapped_modality] = str(file.absolute())

                for modality in modalities:
                    mapped_modality = mapper[modality]
                    if mapped_modality not in available_modalities:
                        subject_dict[mapped_modality] = None

                subject_dict['surv'] = surv
                subject_dict['event'] = event
                subject_dict['group'] = group
                brats_mapping.append(subject_dict)
        combined_mappings = brats_mapping + picture_mapping
        return combined_mappings


