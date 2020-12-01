##################################################
## Trains ResNet or (simple) 3D CNN given parameters. Logs results to MLflow. 
## Logs training to a log file. Quite a lot of stuff is hardcoded such as 
## MLflow tracking URL and various paths. Scripts is highly dependend of
## preprocessed files.
##################################################
## Author: Assil Jwair
## Version: 7.0
## Email: assil.jw@gmail.com
##################################################

##################Imports#######################
import modules.resnet_3d as resnet
from pathlib import Path
import numpy as np
from modules.mapper import FileMapper as fm
from modules.dataset import Dataset
from modules.dataset_image_only import Dataset as dataset_2
from modules.cnn_model import CNNModel
from torch.utils.data import DataLoader
import torchtuples as tt
from pycox.models import LogisticHazard
import pickle
import random
from modules.mlflow_logger import Logger
from modules.utils import getPatientIdentifier, collate_fn, returnMontage, get_optimizer
import sys
import argparse
import datetime
from pycox.evaluation import EvalSurv
from pycox.utils import kaplan_meier
import logging
import logging.config
import pytz
import torch
##################Imports#######################

##### Arguments
parser = argparse.ArgumentParser(prog='ResNet and CNN Trainer',
                                 description='Runs ResNet or CNN model with Logistic Hazard',
                                 epilog='by Assil Jwair')
parser.add_argument('--mri_path',
                    metavar='p',
                    type=str,
                    help='the path to the stored MRIs',
                    default='./data/preoperative_no_norm')
parser.add_argument('--model_depth',
                    metavar='d',
                    type=int,
                    help='Depth of ResNet Model (10, 18, 50)',
                    default=10)
parser.add_argument('--groups',
                    metavar='g',
                    type=int,
                    help='Number of Logistic groups (10, 15)',
                    default=10)
parser.add_argument('--batch_size',
                    metavar='b',
                    type=int,
                    help='batch size',
                    default=128)
parser.add_argument('--optimizer',
                    metavar='o',
                    type=str,
                    help='Optimizer to use (Adam, AdamW, AdamWR)',
                    default='Adam')
parser.add_argument('--normalization',
                    metavar='n',
                    type=str,
                    help='Normalization method to use (zscore, whitestripe, gmm)',
                    default='zscore')
parser.add_argument('--patience',
                    metavar='es',
                    type=int,
                    help='Patience for EarlyStopping',
                    default=5)
parser.add_argument('--model_type',
                    metavar='model_type',
                    type=str,
                    help='ResNet or CNN',
                    default='ResNet')

args = parser.parse_args()

mri_path_ = args.mri_path
model_depth_ = args.model_depth
groups_ = args.groups
batch_size_ = args.batch_size
optimizer_ = get_optimizer(args.optimizer)
normalization_ = args.normalization
patience_ = args.patience
model_type_ = args.model_type
##### Arguments

class Train:
    def __init__(self, mri_path, model_depth, groups, batch_size, optimizer, normalization, patience, logger, mode, optimizer_param=args.optimizer):
        self.mri_path = mri_path
        self.preop_patients = []
        self.model_depth = model_depth
        self.cuts = groups
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.callback = tt.cb.EarlyStopping(patience=patience)
        self.verbose = True
        self.filter_ids = './data/pickles_jsons/filter_ids_v2_all.pkl'
        self.normalized = True
        self.normalization = normalization
        self.dataset_filtered = None
        self.mlflow_url = "http://172.17.0.4:5000"
        self.model = None
        self.logger = logger
        self.mode = mode

        if self.cuts == 10:
            self.id_mapping = './data/pickles_jsons/id_surv_mapping_10_groups.json'
        else:
            self.id_mapping = './data/pickles_jsons/id_surv_mapping.json'
        for path in Path(mri_path).glob('BMIAXNA*'):
            self.preop_patients.append(path)
        self.logger.info("Number of patients: {}\n".format(len(self.preop_patients)))
        self.optimizer_param = optimizer_param
        self.params = {"normalize": "{}".format(self.normalization),
                            "optimizer": "{}".format(self.optimizer_param),
                            "model_depth": self.model_depth,
                            "dropout": 50,
                            "groups": self.cuts,
                            "flipped": True,
                            "del_sequence": True}
        self.logger.info("Model parameters: {}\n".format(self.params))

    def create_dataset(self):
        """
        Function responsible for creating the test, train and validation PyTorch Dataset class.
        """
        mapper_class = fm(self.preop_patients, self.id_mapping, normalized=self.normalized)
        dataset = mapper_class.generate_mapping()
        with open(self.filter_ids, 'rb') as file:
            filter_ids = pickle.load(file)
        dataset_filtered = [entry for entry in dataset if entry['ENT'] is not None]
        self.dataset_filtered = [entry for entry in dataset_filtered if entry['id'] not in filter_ids]
        random.seed(4)
        random.shuffle(self.dataset_filtered)
        train_dataset = Dataset(self.dataset_filtered, phase='train', normalize=self.normalization)
        val_dataset = Dataset(self.dataset_filtered, phase='val', normalize=self.normalization)
        test_dataset = dataset_2(self.dataset_filtered, phase='test', normalize=self.normalization)
        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(self):
        """
        Function responsible for creating the test, train and validation PyTorch Dataloaders.
        """        
        train_dataset, val_dataset, test_dataset = self.create_dataset()
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=6,
                                  shuffle=False,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=6, shuffle=False,
                                collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False)

        return train_loader, val_loader, test_loader

    def train_model(self):
        """
        Function responsible for creating the model and training it.
        Also uses test dataset to predict survival after training the model.
        """
        self.logger.info("Creating Model..\n")
        self.logger.info("Number of groups: {}\n".format(self.cuts))
        if self.cuts == 15:
            cuts = np.array([0., 48., 82., 126., 184., 231., 279., 330., 383., 436., 507., 633., 764., 1044., 1785.])

        elif self.cuts == 10:
            cuts = np.array([0., 70., 134., 213., 290., 365., 450., 623., 829., 1785.])

        self.logger.info("Generating model..\n")
        if self.mode == 'ResNet':
            net = resnet.generate_model(model_depth=self.model_depth,
                                    n_classes=self.cuts,
                                    n_input_channels=4,
                                    shortcut_type='B',
                                    conv1_t_size=7,
                                    conv1_t_stride=1,
                                    no_max_pool=False,
                                    widen_factor=1.0)
        else:
            net = CNNModel(self.cuts)

        self.logger.info("Creating DataLoaders..\n")
        train_loader, val_loader, test_loader = self.create_dataloaders()
        self.logger.info("Creating DataLoaders Done\n")
        model = LogisticHazard(net, self.optimizer, duration_index=cuts, device=0)
        num_epochs = 100
        self.logger.info("Number of epochs: {}\n".format(num_epochs))
        self.logger.info("Begin Training..\n")
        log = model.fit_dataloader(train_loader, num_epochs, [self.callback], self.verbose,
                                   val_dataloader=val_loader)
        self.logger.info("Training Done\n")
        currentDT = datetime.datetime.now()
        model.save_model_weights('./Results/{}_{}.pt'.format(self.mode, currentDT.strftime("%Y-%m-%d_%H-%M-%S")))
        self.logger.info("Predicting with Test Dataset..\n")
        predictions = model.predict_surv_df(test_loader)
        self.logger.info("Predicting with Test Dataset done\n")
        self.logger.info("Predicting with interpolated Test Dataset..\n")
        predictions_interpolated = model.interpolate(self.cuts).predict_surv_df(test_loader)
        self.logger.info("Predicting with interpolated Test Dataset.. done\n")
        return log, predictions, predictions_interpolated

    def trainer(self):
        """
        Function responsible for running the training. 
        """
        log, predictions, predictions_interpolated = self.train_model()
        self.logger.info("Logging Trained Model to MLFlow..\n")
        logger = Logger(self.mlflow_url, "3D_CNN")
        logger.log_metrics(log, "{}".format(self.mode), self.params)
        self.evaluate(predictions=predictions, predictions_interpolated=predictions_interpolated)
        del log
        del predictions
        del predictions_interpolated
        torch.cuda.empty_cache()

    def evaluate(self, predictions, predictions_interpolated):
        """
        Function responsible for computing evaluation metrics (C-index, Brier score). 
        Also responsible for logging said metrics to MLflow using my Logger module.
        
        """
        self.logger.info("Evaluating Model Strength.. \n")
        targets = dataset_2(self.dataset_filtered, phase='test', targets=True)[0][0]
        events = dataset_2(self.dataset_filtered, phase='test', targets=True)[0][1]
        survs = dataset_2(self.dataset_filtered, phase='test', targets=True)[0][2]
        censored = []
        for surv in events:
            if surv == 1:
                censored.append(False)
            else:
                censored.append(True)
        self.logger.info("Calculating concordance and brier score\n")
        ev = EvalSurv(predictions_interpolated, survs, events, 'km')
        concordance = ev.concordance_td()
        time_grid = np.linspace(0, survs.max())
        integrated_brier_score = ev.integrated_brier_score(time_grid)
        self.logger.info("Concordance: {}, Integrated Brier Score: {}\n".format(concordance, integrated_brier_score))

        logger = Logger(self.mlflow_url, "CNN_Predictions")
        logger.log_predictions(predictions, "{}".format(self.mode),  concordance,  integrated_brier_score, self.params, targets, self.cuts)
        self.logger.info("Logged Evaluation metrics to MLFlow\n")


if __name__ == '__main__':
    tz_amsterdam = pytz.timezone('Europe/Amsterdam')
    currentDT = datetime.datetime.now(tz_amsterdam).strftime("%Y-%m-%d_%H-%M")
    log_file_path = './logs/{}-{}.txt'.format(currentDT, model_type_)
    if not Path(log_file_path).is_file():
        with open(log_file_path, 'w') as file:
            pass
    logging.basicConfig(filename=log_file_path,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logging.info("Running {} Trainer".format(model_type_))

    logger_ = logging.getLogger('{}'.format(model_type_))

    try:
        trainer_class = Train(mri_path_, model_depth_, groups_, batch_size_, optimizer_, normalization_, patience_, logger_, model_type_)
        trainer_class.trainer()
    except Exception as e:
        logger_.error('Error at %s', 'training', exc_info=e)


