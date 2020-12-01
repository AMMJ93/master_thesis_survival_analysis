import sys
import os
import nibabel as nib
from pathlib import Path
import numpy as np
from modules.normalization import main as norm_main
from tqdm import tqdm
import pickle
from modules.mapper import FileMapper as fm
import random
from shutil import copy2, copytree

def save_image(path_str, img=False, return_new_path=False):
    path = Path(path_str)
    path_parts = list(path.parts)
    for i, part in enumerate(path_parts):
        if part == 'preoperative_no_norm':
            path_parts[i] = 'preoperative_zscore'
    new_path = Path('/'.join(path_parts)[1:])
    Path(new_path.parent).mkdir(parents=True)
    if return_new_path:
        return new_path
    else:
        nib.save(img, new_path)


def convert_images(dataset):
    sys.stdout.write('Starting Copy\n')
    modalities = ['T1c', 'T1w','FLR', 'ENT', 'T2w']
    sys.stdout.write('Dataset Length: {}\n'.format(len(dataset)))
    for entry in tqdm(dataset):
        for modality in modalities:
            if modality != 'ENT':
                path = entry[modality]
                nifti_img = nib.load(entry[modality])
                min_ = nifti_img.get_fdata().min()
                max_ = nifti_img.get_fdata().max()
                if min_ == -0 and max_ == -0:
                    new_path = save_image(path, return_new_path=True)
                    copy2(path, new_path)
                else:
                    image_nib = norm_main(nifti_img, contrast=modality, method='zscore')
                    save_image(path, img=image_nib)
            elif modality == 'ENT':
                path = entry[modality]
                new_path = save_image(path, return_new_path=True)
                copy2(path, new_path)

def main():
    preop_patients = []
    for path in Path('./data/preoperative_no_norm').glob('BMIAXNA*'):
        preop_patients.append(path)
    id_mapping = './data/pickles_jsons/id_surv_mapping_10_groups.json'
    mapper_class = fm(preop_patients, id_mapping, normalized=True)
    dataset = mapper_class.generate_mapping()
    with open('./data/pickles_jsons/filter_ids_v2_all.pkl', 'rb') as file:
        filter_ids = pickle.load(file)
    dataset_filtered = [entry for entry in dataset if entry['ENT'] is not None]
    dataset_filtered = [entry for entry in dataset_filtered if entry['id'] not in filter_ids]
    random.seed(4)
    random.shuffle(dataset_filtered)
    convert_images(dataset_filtered)


if __name__ == "__main__":
    main()

