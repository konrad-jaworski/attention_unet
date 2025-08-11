import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import random
import os
import json

class Compose3D:
    """
    Class handling number of transformation applied to our data
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image,label=None):
        for t in self.transforms:
            image,label = t(image,label)
        return image, label

class RandomFlip3D:
    def __init__(self, axes=(0, 1, 2), p=0.5):
        self.axes = axes  # which axes to possibly flip: 0=H, 1=W, 2=D
        self.p = p

    def __call__(self, image, label=None):
        for axis in self.axes:
            if random.random() < self.p:
                image = torch.flip(image, dims=[axis + 1])  # +1 to skip channel dim
                if label is not None:
                    label = torch.flip(label, dims=[axis + 1])
        return image, label
    
class AddGaussianNoise3D:
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, image,label=None):
        noise = torch.randn_like(image) * self.std + self.mean
        aug_img=image + noise
        return aug_img,label

# Definite behaviour for thermogram sequences
class IntensityShift3D:
    def __init__(self, shift_range=(-0.1, 0.1), scale_range=(0.9, 1.1)):
        self.shift_range = shift_range
        self.scale_range = scale_range

    def __call__(self, image,label=None):
        shift = random.uniform(*self.shift_range)
        scale = random.uniform(*self.scale_range)
        aug_img=image * scale + shift
        return aug_img,label
    
class NormalizeTo01:
    def __call__(self,image,label=None):
        min_val=image.min()
        max_val=image.max()
        image=(image-min_val)/(max_val-min_val+1e-5)
        return image, label