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
    print(x.shape,y)

    # train_dataset, rotated_train_dataset = get_data_generator(dataset.x_train, dataset.y_train, config.batch_size)
    # test_dataset, rotated_test_dataset = get_data_generator(dataset.x_test, dataset.y_test, config.batch_size)



    



