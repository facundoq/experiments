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

def run_all_dataset(model,dataset,config,rotations,batch_size=256):
    dataset= ImageDataset(dataset.x_test,dataset.y_test, rotation=True)
    layer_invariances = eval_invariance_measure(dataset, model, config, rotations, batch_size)
    return [layer_invariances],[0]

def run(model,dataset,config,rotations,batch_size=256):
    x = dataset.x_test
    y=dataset.y_test
    y_ids=y.argmax(axis=1)
    classes=np.unique(y_ids)
    classes.sort()

    class_layer_invariances=[]
    for i, c in enumerate(classes):
        # logging.debug(f"Evaluating invariances for class {c}...")
        ids=np.where(y_ids==c)
        ids=ids[0]
        x_class,y_class=x[ids,:],y[ids]
        class_dataset=ImageDataset(x_class,y_class,rotation=True)
        layer_invariances=eval_invariance_measure(class_dataset, model, config, rotations, batch_size)
        class_layer_invariances.append(layer_invariances)
    return class_layer_invariances,classes

def eval_invariance_measure(dataset,model,config,rotations,batch_size):
    layer_invariances_baselines=get_baseline_variance_class(dataset,model,config,rotations,batch_size)
    layer_invariances = eval_invariance_class(dataset, model, config, rotations,batch_size)
    normalized_layer_invariances = calculate_invariance_measure(layer_invariances_baselines,layer_invariances)
    return normalized_layer_invariances

def calculate_invariance_measure(layer_baselines, layer_measures):
    eps=0
    measures = []  # coefficient of variations
    for layer_baseline, layer_measure in zip(layer_baselines, layer_measures):
        normalized_measure = layer_measure[layer_baseline > eps] / layer_baseline[layer_baseline > eps]
        measures.append(normalized_measure)
    return measures

def get_baseline_variance_class(dataset,model,config,rotations,batch_size):
    n_intermediates = model.n_intermediates()
    baseline_variances = [RunningMeanAndVariance() for i in range(n_intermediates)]

    for i, r in enumerate(rotations):
        degrees = (r - 1, r + 1)
        # logging.debug(f"    Rotation {degrees}...")
        dataset.update_rotation_angle(degrees)
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False, num_workers=0,drop_last=True)
        # calculate std for all examples and this rotation
        dataset_invariance_measure=get_dataset_invariance_measure(model,dataloader,config)
        #update the mean of the stds for every rotation
        # each std is intra rotation/class, so it measures the baseline
        # std for that activation
        for j,m in enumerate(dataset_invariance_measure):
            baseline_variances[j].update(m.std())
    mean_baseline_variances=[b.mean() for b in baseline_variances]
    return mean_baseline_variances

def get_dataset_invariance_measure(model,dataloader,config):
    n_intermediates = model.n_intermediates()
    invariance_measure = [RunningMeanAndVariance() for i in range(n_intermediates)]
    for x,y_true in dataloader:
        if config.use_cuda:
            x = x.cuda()
        with torch.no_grad():
            y, intermediates = model.forward_intermediates(x)
            for j, intermediate in enumerate(intermediates):
                flat_activations=transform_activations(intermediate)
                for h in range(flat_activations.shape[0]):
                    invariance_measure[j].update(flat_activations[h,:])
    return invariance_measure





#returns a list of RunningMeanAndVariance objects,
# one for each intermediate output of the model.
#Each RunningMeanAndVariance contains the mean and std of each intermediate
# output over the set of rotations
def eval_invariance_class(dataset,model,config,rotations,batch_size):
    n_intermediates = model.n_intermediates()
    layer_invariances= [RunningMeanAndVariance() for i in range(n_intermediates)]
    n = len(dataset)
    batch_ranges=[ range(i,i+batch_size) for i in range(n//batch_size)]

    for batch_range in batch_ranges:
        batch_invariance=BatchInvarianceMeasure(batch_size,n_intermediates)
        for r in rotations:
            degrees = (r - 1, r + 1)
            dataset.update_rotation_angle(degrees)
            x,y_true=dataset.get_batch(batch_range)
            if config.use_cuda:
                x = x.cuda()
            with torch.no_grad():
                y, batch_activations= model.forward_intermediates(x)
                batch_activations=[transform_activations(a) for a in batch_activations]
                batch_invariance.update(batch_activations)
        batch_invariance.update_global_measures(layer_invariances)

    mean_layer_invariances = [b.mean() for b in layer_invariances]
    return mean_layer_invariances

class BatchInvarianceMeasure:
    def __init__(self,batch_size,n_intermediates):
        self.batch_size=batch_size
        self.n_intermediates=n_intermediates
        self.batch_stats = [[RunningMeanAndVariance() for i in range(batch_size)] for j in range(n_intermediates)]
        self.batch_stats = np.array(self.batch_stats)

    def update(self,batch_activations):
        for i, layer_activations in enumerate(batch_activations):
            for j in range(layer_activations.shape[0]):
                self.batch_stats[i, j].update(layer_activations[j, :])

    def update_global_measures(self,dataset_stats):
        for i in range(self.n_intermediates):
            mean_invariance=dataset_stats[i]
            for j in range(self.batch_size):
                mean_invariance.update(self.batch_stats[i, j].std())


def transform_activations(activations_gpu):
    activations = activations_gpu.detach().cpu().numpy()

    # if conv average out spatial dims
    if len(activations.shape) == 4:
        n, c, w, h = activations.shape
        flat_activations = np.zeros((n, c))
        for i in range(n):
            flat_activations[i, :] = activations[i, :, :, :].mean(axis=(1, 2))
        assert (len(flat_activations.shape) == 2)
    else:
        flat_activations = activations

    return flat_activations




def plot_class_outputs(class_id, cvs, names,model_name,dataset_name,savefig):
    n = len(names)
    f, axes = plt.subplots(1, n, dpi=150)
    max_cv = max([cv.max() for cv in cvs])

    for i, (cv, name) in enumerate(zip(cvs, names)):
        ax = axes[i]
        ax.axis("off")

        cv = cv[:, np.newaxis]
        mappable=ax.imshow(cv,vmin=0,vmax=max_cv,cmap='inferno')
        #mappable = ax.imshow(cv, cmap='inferno')
        ax.set_title(name, fontsize=7)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.suptitle(f"sigma for class {class_id}")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(mappable, cax=cbar_ax)
    if savefig:
        path=os.path.join("plots","invariance",f"invariance_{model_name}_{dataset_name}_class{class_id}.png")
        plt.savefig(path)
    plt.show()

def plot(all_stds,model,dataset_name,classes,savefig=False):

    for i,c in enumerate(classes):
        stds=all_stds[i]
        plot_class_outputs(c, stds, model.intermediates_names(),model.name,dataset_name,savefig)

