import pickle
from pathlib import Path
from modules.mapper import FileMapper as fm
from modules.dataset import Dataset


def main():
    preop_patients = []
    for path in Path('./data/preoperative').rglob('BMIAXNA*'):
        preop_patients.append(path)
    mapper_class = fm(preop_patients, './data/pickles_jsons/id_surv_mapping.json')
    dataset = mapper_class.generate_mapping()
    dataset_filtered = [entry for entry in dataset if entry['ENT'] is not None]
    train_dataset = Dataset(dataset_filtered, phase='train')
    val_dataset = Dataset(dataset_filtered, phase='val')
    filter_ids = []
    for data in train_dataset:
        if 'BMIAXNAT' in data:
            filter_ids.append(data)

    for data in val_dataset:
        if 'BMIAXNAT' in data:
            filter_ids.append(data)

    with open('./data/pickles_jsons/filter_ids.pkl', 'wb') as file:
        pickle.dump(filter_ids, file)


if __name__ == "__main__":
    main()