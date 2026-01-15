import os
import cv2
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None, size=(256,256), N=7,maskPath=None):
        self.data_dir = data_dir
        self.maskPath = maskPath
        self.transform = transforms
        self.size      = size

        self.lst_data = os.listdir(self.data_dir)
        self.lst_data.sort()

        if maskPath is not None:
            self.mask_lst = os.listdir(self.maskPath)
            self.mask_lst.sort()
        

        self.to_tensor = ToTensor()
    
    def __len__(self):
        return len(self.lst_data)
    
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.data_dir, self.lst_data[index]), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        

        # Image Scaling
        if img.dtype == np.uint8:
            img = img / 255.0

        data = {"gt": img.copy(), "inputs":img.copy()} # Label

        if self.maskPath is not None:
            mask = cv2.imread(os.path.join(self.maskPath, self.mask_lst[index]), cv2.IMREAD_COLOR)
            mask = mask / 255.0
            data["mask"] = mask

        # If Task is Inpainting
        # Data Transforming
        if self.transform:
            data = self.transform(data)
        
        data = self.to_tensor(data)

        return data

class ToTensor(object):

    # Make Image [B, H, W, C] to [B, C, H, W]
    # Make Image Numpy Array to Tensor

    def __call__(self, data):

        for key, value in data.items():
            if key == "inputs" or key == "real" or key=="indexMap":
                value = value.transpose((2, 0, 1)).astype(np.float32)
                data[key] = torch.from_numpy(value)
            else:
                data[key] = torch.from_numpy(np.array(value))

        return data



if __name__ == "__main__":
    testPath = "../Data/FFHQ"
    testDataset = Dataset(data_dir=testPath, patchSize=16, size=(256,256), N=7, task="outpainting", paintingSize=2)

    print(f"{testDataset.patchWNumber}, {testDataset.patchIndex}, {np.shape(testDataset.patchIndex)}")
  