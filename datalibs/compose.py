import cv2
import numpy as np
import cupy as cp


class Resize(object):

    # ReSizing Data

    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        inputs = data["inputs"]
        real = data["gt"]

        data["inputs"] = cv2.resize(inputs, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)      
        data["gt"] = cv2.resize(real, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_LINEAR)
        
        return data

class Normalization(object):
    
    # Normalized Data

    def __init__(self, mean=0.5, std= 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):

        real = data["gt"]

        data["gt"] = (real - self.mean) / self.std

        return data