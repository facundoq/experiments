import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
logging.getLogger().setLevel(logging.DEBUG)

verbose=False

import torch
use_cuda=torch.cuda.is_available()
import pytorch
from pytorch import dataset as ptd

import os
import datasets
import torch.optim as optim
import numpy as np

dataset_name="cifar10"

(x_train, y_train), (x_test, y_test), input_shape,num_classes = datasets.get_data(dataset_name)
dataset=ptd.ClassificationDataset(dataset_name,x_train,x_test,y_train,y_test,num_classes,input_shape)


print(f"Training with dataset {dataset_name}.")
if verbose:
    print('x_train shape:', x_train.shape,x_train.dtype)
    print('x_test shape:', x_test.shape,x_test.dtype)
    print('y_train shape:', y_train.shape,y_train.dtype)
    print('y_test shape:', y_test.shape,y_test.dtype)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print("Classes ", np.unique(y_train.argmax(axis=1)))
    print("min class/max class:", y_train.min(),y_train.max())



from pytorch.experiment import models

model_name="SimpleConv"
model, optimizer, rotated_model, rotated_optimizer = models.get_model(model_name,dataset,use_cuda)
if verbose:
    print(model)
    print(rotated_model)


from pytorch.experiment import rotation
batch_size = 64

def get_epochs(dataset,model):
    if model=="SimpleConv":
        epochs={'cifar10':70,'mnist':5,'fashion_mnist':12,'cluttered_mnist':10,'lsa16':50,'mnist_rot':5,'pugeault':15}
        rotated_epochs={'cifar10':120,'mnist':15,'fashion_mnist':60,'cluttered_mnist':30,'lsa16':100,'mnist_rot':5,'pugeault':40}
    elif model=="AllConv":
        epochs={'cifar10':5,'mnist':2,'fashion_mnist':12,'cluttered_mnist':10,'lsa16':50,'mnist_rot':5,'pugeault':15}
        rotated_epochs={'cifar10':100,'mnist':5,'fashion_mnist':60,'cluttered_mnist':30,'lsa16':100,'mnist_rot':5,'pugeault':40}
    else:
        raise ValueError(f"Invalid model name: {model}")
    return epochs[dataset],rotated_epochs[dataset]

epochs,rotated_epochs=get_epochs(dataset.name,model_name)

config=rotation.TrainRotatedConfig(batch_size=batch_size,
                   epochs=epochs,rotated_epochs=rotated_epochs,
                   pre_rotated_epochs=0, optimizer=optimizer,rotated_optimizer=rotated_optimizer,
                  use_cuda=use_cuda)

scores=rotation.run(config,model,rotated_model,dataset,plot_accuracy=True,save_plots=True)
rotation.print_scores(scores)

save_model=True
if save_model:
    rotation.save_models(dataset,model,rotated_model,scores,config)

