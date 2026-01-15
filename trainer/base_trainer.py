import json, os
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist


from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


from datalibs import EEGPrepDataset
from datalibs.compose import *

class BaseTrainer():
    def __init__(self, ckpt_dir:str, log_dir:str, batch_size:int, sharedFilePath:str, num_workers:int=6):
        self.ckpt_dir       = ckpt_dir
        self.log_dir        = log_dir
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.sharedFilePath = sharedFilePath
    
    def __checkDirectory__(self):

        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
            
        if not os.path.exists(os.path.join(self.ckpt_dir, self.name)):
            os.mkdir(os.path.join(self.ckpt_dir, self.name))

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def initialize(self, gpu, size):
        # For Debugging If, You want. using below it
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ['NCCL_DEBUG_SUBSYS']="ALL"
        # For Online Learning, In My enviroment, can't using it. if want to use chang, init_method.
        # os.environ["MASTER_ADDR"] = "127.0.0.1"
        # os.environ["MASTER_PORT"] = "29500"

        os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
        os.environ['NCCL_IB_DISABLE']= '1'
        os.environ['LOCAL_RANK'] = str(gpu)
        torch.cuda.set_device(gpu)

        # Setting Model with Json Parsing
        dist.init_process_group(backend='nccl', init_method='file://'+self.sharedFilePath+'/sharedfile', rank=gpu, world_size=size)
        # raise ValueError
    
    def makeDatasets(self, eeg_train_path, eeg_test_path, eeg_val_path, img_path, img_size, mean=None, std=None, min_value=0, max_value = 1, ddp=True):
        self.trasnform_train = transforms.Compose([
            Resize((img_size,img_size)),
            Normalization(mean, std) if mean is not None and std is not None else Normalization(),
            Scaling(min_value, max_value)
        ])
        self.trasnform_valid = transforms.Compose([
            Resize((img_size,img_size)),
            Normalization(mean, std) if mean is not None and std is not None else Normalization(),
            Scaling(min_value, max_value)
        ])
        self.trasnform_test = transforms.Compose([
            Resize((img_size,img_size)),
            Normalization(mean, std) if mean is not None and std is not None else Normalization(),
            Scaling(min_value, max_value)
        ])
        self.train_dataset = EEGPrepDataset(eeg_pre_path = eeg_train_path,  eeg_data_path = img_path, transforms=self.trasnform_train, img_size = img_size)
        self.valid_dataset = EEGPrepDataset(eeg_pre_path = eeg_val_path,  eeg_data_path = img_path, transforms=self.trasnform_valid, img_size = img_size)
        self.test_dataset  = EEGPrepDataset(eeg_pre_path = eeg_test_path,  eeg_data_path = img_path, transforms=self.trasnform_test, img_size = img_size)

        if ddp:
            self.train_dataset_sampler = DistributedSampler(self.train_dataset, drop_last=True)
            self.valid_dataset_sampler = DistributedSampler(self.valid_dataset, drop_last=True)
            self.test_dataset_sampler  = DistributedSampler(self.test_dataset, drop_last=True)

            self.loader_train       = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, sampler=self.train_dataset_sampler)
            self.loader_valid       = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, sampler=self.valid_dataset_sampler)
            self.loader_test        = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, sampler=self.test_dataset_sampler) 
        else:
            self.loader_train       = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
            self.loader_valid       = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,)
            self.loader_test        = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True,)

    def makeTensorBoard(self):
        self.summaryWriter = SummaryWriter(log_dir=self.log_dir)


    def json_load(self, path:str):
        with open(path, "r") as jsonFile:
            jsonDict = json.load(jsonFile)
        return jsonDict

