##################################################
## Dirty way to map IDs to their corresponding phase (train, test, validation). 
## This mapping is needed to ensure the clinical dataset model also uses the same
## patients for the data splits (train, test, validation). Outputs pickle file. 
##################################################
## Author: Assil Jwair
## Version: 1.0
##################################################

from pathlib import Path
from modules.mapper import FileMapper as fm
from modules.dataset import Dataset
import torchtuples as tt
import pickle
import random
import sys


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

    train = dataset_filtered[:int(len(dataset_filtered) * 0.7)]
    test = dataset_filtered[int(len(dataset_filtered) * 0.8):]
    val = dataset_filtered[int(len(dataset_filtered) * 0.7):int(len(dataset_filtered) * 0.8)]

    val_ids = [entry['id'] for entry in val]
    train_ids = [entry['id'] for entry in train]
    test_ids = [entry['id'] for entry in test]
    ids_per_phase = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    with open('./data/pickles_jsons/ids_per_phase.pkl', 'wb') as file:
        pickle.dump(ids_per_phase, file)

        
if __name__ == "__main__":
    main()

