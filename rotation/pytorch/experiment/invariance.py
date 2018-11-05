import matplotlib.pyplot as plt
from pytorch.experiment.utils import RunningMeanAndVariance
from collections import namedtuple
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import os
import progressbar
import logging

from pytorch.dataset import get_data_generator

from pytorch.dataset import ImageDataset

def run(model,dataset,config,n_rotations):
    x = dataset.x_test
    y=dataset.y_test
    y_ids=y.argmax(axis=1)
    classes=np.unique(y_ids)
    classes.sort()
    batch_size=256

    rotations=np.linspace(-179,180,n_rotations,endpoint=False)
    all_cvs=[]
    for i, c in enumerate(classes):
        # logging.debug(f"Evaluating invariances for class {c}...")
        ids=np.where(y_ids==c)
        # ids=ids[0]
        # ids=np.array(ids[:batch_size])
        x_class,y_class=x[ids,:],y[ids]
        class_dataset=ImageDataset(x_class,y_class,rotation=True)
        class_baselines=get_baseline_variance_class(model,class_dataset,config,rotations)
        class_stats= eval_invariance_class(class_dataset,model,config,rotations)
        cvs = calculate_coefficient_of_variation(class_baselines,class_stats)
        all_cvs.append(cvs)

    return all_cvs,classes

def transform_activations(activations):
    intermediate = activations.detach().cpu().numpy()
    # average out batch dim
    intermediate = intermediate.mean(axis=0)
    # if conv average out spatial dims
    if len(intermediate.shape) == 3:
        intermediate = intermediate.mean(axis=(1, 2))
        assert (len(intermediate.shape) == 1)
    return intermediate

def get_baseline_variance_class(model,dataset,config,rotations):
    n_intermediates = model.n_intermediates()
    std_mean = [RunningMeanAndVariance() for i in range(n_intermediates)]

    for i, r in enumerate(rotations):
        degrees = (r - 1, r + 1)
        # logging.debug(f"    Rotation {degrees}...")
        dataset.update_rotation_angle(degrees)
        dataloader=DataLoader(dataset,batch_size=128,shuffle=False, num_workers=1)
        means_rotation = [RunningMeanAndVariance() for i in range(n_intermediates)]
        # calculate std for all examples and this rotation
        for x,y_true in dataloader:
            if config.use_cuda:
                x = x.cuda()
            with torch.no_grad():
                y, intermediates = model.forward_intermediates(x)
                for i, intermediate in enumerate(intermediates):
                    flat_activations=transform_activations(intermediate)
                    # update running mean for this layer
                    means_rotation[i].update(flat_activations)

        #update the mean of the stds for every rotation
        # each std is intra rotation/class, so it measures the baseline
        # std for that activation
        for i,m in enumerate(means_rotation):
            std_mean[i].update(m.std())
    return std_mean

def plot_class_outputs(class_id,cvs,names):
    n=len(names)
    f,axes=plt.subplots(1,n,dpi=150)
    max_cv=max([cv.max() for cv in cvs])

    for i,(cv,name) in enumerate(zip(cvs,names)):
        ax=axes[i]
        ax.axis("off")

        cv=cv[:,np.newaxis]
        #mappable=ax.imshow(cv,vmin=0,vmax=max_cv,cmap='jet')
        mappable = ax.imshow(cv, cmap='inferno')
        ax.set_title(name,fontsize=7)

         #logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.suptitle(f"sigma for class {class_id}")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(mappable, cax=cbar_ax)
    plt.show()


def calculate_coefficient_of_variation(class_baselines,class_stats):
    cvs = []  # coefficient of variations
    for baseline,stat in zip(class_baselines,class_stats):
        std = stat.std()
        baseline_std = baseline.mean()
        std[baseline_std > 0] /= baseline_std[baseline_std > 0]
        cvs.append(std)
    return cvs

# train_dataset, rotated_train_dataset = get_data_generator(dataset.x_train, dataset.y_train, config.batch_size)
# test_dataset, rotated_test_dataset = get_data_generator(dataset.x_test, dataset.y_test, config.batch_size)

#returns a list of RunningMeanAndVariance objects,
# one for each intermediate output of the model.
#Each RunningMeanAndVariance contains the mean and std of each intermediate
# output over the set of rotations
def eval_invariance_class(dataset,model,config,rotations):
    n_intermediates = model.n_intermediates()
    running_means = [RunningMeanAndVariance() for i in range(n_intermediates)]

    for i,r in enumerate(rotations):
        degrees = (r - 1, r + 1)
        # logging.debug(f"    Rotation {degrees}...")
        dataset.update_rotation_angle(degrees)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
        for x, y_true in dataloader:
            if config.use_cuda:
                x = x.cuda()
            with torch.no_grad():
                y, intermediates = model.forward_intermediates(x)
                for i, intermediate in enumerate(intermediates):
                    flat_activations = transform_activations(intermediate)
                    running_means[i].update(flat_activations)

    return running_means


def plot_class_outputs(class_id, cvs, names):

    n = len(names)
    f, axes = plt.subplots(1, n, dpi=100)
    max_cv = max([cv.max() for cv in cvs])

    for i, (cv, name) in enumerate(zip(cvs, names)):
        ax = axes[i]
        ax.axis("off")

        cv = cv[:, np.newaxis]
        # mappable=ax.imshow(cv,vmin=0,vmax=max_cv,cmap='jet')
        mappable = ax.imshow(cv, cmap='jet')
        ax.set_title(name, fontsize=7)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.suptitle(f"sigma for class {class_id}")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(mappable, cax=cbar_ax)
    plt.show()

def plot(all_stds,model,classes):

    for i,c in enumerate(classes):
        stds=all_stds[i]
        plot_class_outputs(c, stds, model.intermediates_names())