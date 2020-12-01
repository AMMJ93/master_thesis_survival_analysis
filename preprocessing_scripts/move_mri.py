from pathlib import Path
from shutil import copy2, copytree
import sys
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(prog='Move preop files',
                                 description='Moves preop files only to ./data/preoperative_no_norm',
                                 epilog='by Assil Jwair')

parser.add_argument('--mri_path',
                    metavar='p',
                    type=str,
                    help='the path to the stored MRIs')

parser.add_argument('--new_path',
                    metavar='np',
                    type=str,
                    help='the path to the new parent folder to store preoperative only')

args = parser.parse_args()

mri_path_ = args.mri_path
new_path_ = args.new_path


def main(mris_path, new_folder):
    preop_directories = []
    for path in Path(mris_path).rglob('preop*'):
        preop_directories.append(path)

    if Path(new_folder).is_dir() == False:
        Path(new_folder).mkdir()
    if Path('../data/preoperative_no_norm').is_dir() == False:
        Path('../data/preoperative_no_norm').mkdir()
    main_path = Path('../data/preoperative_no_norm')
    print("Number of patients: {}".format(len(preop_directories)))
    for i, path in tqdm(enumerate(preop_directories)):
        scan_dir = path.parts[9]
        scan_path = main_path.joinpath(scan_dir)
        copytree(path, scan_path)


if __name__ == "__main__":
    main(mri_path_, new_path_)

