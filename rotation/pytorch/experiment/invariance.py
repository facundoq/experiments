#import matplotlib.pyplot as plt

from collections import namedtuple
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import os
import progressbar

from pytorch.dataset import get_data_generator

from pytorch.dataset import ImageDataset

def analyze_outputs(model,dataset,config):
    image_dataset = ImageDataset(dataset.x_test, dataset.y_test)
    a=[1,2,3]
    x,y=image_dataset.get_batch(a)
    print("batch",  x.shape,y)

    classes=np.unique(dataset.y_test.argmax(axis=1))
    classes.sort()
    print(classes)

    def hook(module,input,output):
        print(f"hook for {module}, output: ",output.shape)

    n_intermediates=14
    for i, c in enumerate(classes):
        running_means = [RunningMeanAndVariance() for i in range(n_intermediates)]
        if config.use_cuda:
            x=x.cuda()
        y2, intermediates = model.forward_intermediates(x)
        for i,intermediate in enumerate(intermediates):
            running_means[i].update(intermediate[i])

    # model.conv.register_forward_hook(hook)
    # model.fc.register_forward_hook(hook)

    print(y2.shape,len(intermediates))

    # train_dataset, rotated_train_dataset = get_data_generator(dataset.x_train, dataset.y_train, config.batch_size)
    # test_dataset, rotated_test_dataset = get_data_generator(dataset.x_test, dataset.y_test, config.batch_size)

class RunningMeanAndVariance:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())