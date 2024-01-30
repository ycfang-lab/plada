import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class VisDA(Dataset):
    def __init__(self, train=True, path=None):
        self.train = train
        if path is None:
            self.path = './data/visda'
        else:
            self.path = path
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])

        self.transform_eval = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize])


        if train:
            self.list_path = os.path.join(self.path, 'train', '')