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
    n=dataset.x_test.shape[0]
    effective_batch_size = min(n, batch_size)
    dataset= ImageDataset(dataset.x_test,dataset.y_test, rotation=True)
    layer_invariances = eval_invariance_measure(dataset, model, config, rotations, effective_batch_size)
    return [layer_invariances],[0]

def run(model,dataset,config,rotations,batch_size=256):
    x = dataset.x_test
    y=dataset.y_test
    y_ids=y.argmax(axis=1)
    classes=np.unique(y_ids)
    classes.sort()

    class_layer_invariances=[]
    # calculate the invariance measure for each class
    for i, c in enumerate(classes):
        # logging.debug(f"Evaluating invariances for class {c}...")
        ids=np.where(y_ids==c)
        ids=ids[0]
        x_class,y_class=x[ids,:],y[ids]
        n=x_class.shape[0]
        class_dataset=ImageDataset(x_class,y_class,rotation=True)
        effective_batch_size=min(n,batch_size)
        layer_invariances=eval_invariance_measure(class_dataset, model, config, rotations, effective_batch_size)
        class_layer_invariances.append(layer_invariances)
    stratified_layer_invariances=eval_stratified_invariance(class_layer_invariances)

    return class_layer_invariances,stratified_layer_invariances,classes

# calculate the mean activation of each unit in each layer over the set of classes
def eval_stratified_invariance(class_layer_invariances):

    layer_class_invariances=[list(i) for i in zip(*class_layer_invariances)]
    layer_invariances=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_invariances]
    return [layer_invariances]

# calculate the invariance measure for a given dataset and model
def eval_invariance_measure(dataset,model,config,rotations,batch_size):
    #baseline invariances
    layer_invariances_baselines=get_baseline_variance_class(dataset,model,config,rotations,batch_size)
    layer_invariances = eval_invariance_class(dataset, model, config, rotations,batch_size)
    normalized_layer_invariances = calculate_invariance_measure(layer_invariances_baselines,layer_invariances)

    # for i in range(len(layer_invariances_baselines)):
    #     print(f"class {i}")
    #     n=len(layer_invariances_baselines[i])
    #     print([ type(layer_invariances_baselines[i][j]) for j in range(n)])
    #     print([type(layer_invariances[i][j]) for j in range(n)])
    #     print([type(normalized_layer_invariances[i][j]) for j in range(n)])


    return normalized_layer_invariances

def calculate_invariance_measure(layer_baselines, layer_measures):
    eps=0
    measures = []  # coefficient of variations

    for layer_baseline, layer_measure in zip(layer_baselines, layer_measures):
        #print(layer_baseline.shape, layer_measure.shape)
        normalized_measure= layer_measure.copy()
        normalized_measure[layer_baseline > eps] /= layer_baseline[layer_baseline > eps]
        both_below_eps=np.logical_and(layer_baseline <= eps,layer_measure <= eps )
        normalized_measure[both_below_eps] = 1
        only_baseline_below_eps=np.logical_and(layer_baseline <= eps,layer_measure > eps )
        normalized_measure[only_baseline_below_eps] = np.inf
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




def plot_class_outputs(class_id, cvs,vmin,vmax, names,model_name,dataset_name,savefig,savefig_suffix):
    n = len(names)
    f, axes = plt.subplots(1, n, dpi=150)


    for i, (cv, name) in enumerate(zip(cvs, names)):
        ax = axes[i]
        ax.axis("off")
        cv = cv[:, np.newaxis]
        mappable=ax.imshow(cv,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")
        #mappable = ax.imshow(cv, cmap='inferno')
        ax.set_title(name, fontsize=5)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.suptitle(f"sigma for class {class_id}")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='blue')
    if savefig:
        image_name=f"invariance_{model_name}_{dataset_name}_{savefig_suffix}_class{class_id}.png"
        path=os.path.join("plots","invariance",image_name)
        plt.savefig(path)
    plt.show()

def pearson_outlier_range(values,iqr_away):
    p50 = np.median(values)
    p75 = np.percentile(values, 75)
    p25 = np.percentile(values, 25)
    iqr = p75 - p25

    range = (p50 - iqr_away * iqr, p50 + iqr_away * iqr)
    return range

def outlier_range_both(rotated_stds,unrotated_stds,iqr_away=5):
    rmin,rmax=outlier_range(rotated_stds,iqr_away)
    umin,umax= outlier_range(unrotated_stds,iqr_away)

    return (max(rmin,umin),min(rmax,umax))

def outlier_range(stds,iqr_away):
    class_values=[np.hstack(class_stds) for class_stds in stds]
    values=np.hstack(class_values)

    pmin,pmax=pearson_outlier_range(values,iqr_away)
    # if the pearson outlier range is away from the max and/or min, use max/or and min instead
    min_stds_all = min([min([std.min() for std in class_stds]) for class_stds in stds])
    max_stds_all = max([max([std.max() for std in class_stds]) for class_stds in stds])
    return ( max(pmin,min_stds_all),min(pmax,max_stds_all))

def plot(all_stds,model,dataset_name,classes,savefig=False,savefig_suffix="",class_names=None,vmax=None):

    vmin=0

    for i,c in enumerate(classes):
        stds=all_stds[i]
        if class_names:
            name=class_names[c]
        else:
            name=str(c)
        plot_class_outputs(name, stds,vmin,vmax, model.intermediates_names(),model.name,
                           dataset_name,savefig,
                           savefig_suffix)
def plot_collapsing_layers(rotated_measures,measures,rotated_model,model,dataset_name,labels):
    rotated_measures_collapsed=collapse_measure_layers(rotated_measures)
    measures_collapsed=collapse_measure_layers(measures)
    n=len(rotated_measures)
    assert(n==len(measures))
    assert (n == len(labels))

    color = plt.cm.hsv(np.linspace(0.1, 0.9, n))


    f,ax=plt.subplots(dpi=min(300,n*15))
    for rotated_measure,measure,label,i in zip(rotated_measures_collapsed,measures_collapsed,labels,range(n)):
        x=np.arange(rotated_measure.shape[0])
        ax.plot(x,rotated_measure,label="rotated_"+label,linestyle="-",color=color[i,:])
        ax.plot(np.arange(measure.shape[0]),measure,label="unrotated_"+label,linestyle="--",color=color[i,:])

    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.5, 1.05))

    plt.show()



def collapse_measure_layers(measures):
    return [np.array([np.mean(layer) for layer in measure]) for measure in measures]



