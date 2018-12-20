import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
logging.getLogger().setLevel(logging.DEBUG)
import torch
use_cuda=torch.cuda.is_available()

# DATASET
from pytorch import dataset as datasets

from pytorch.experiment import models,rotation
import pytorch_models

import pytorch.experiment.utils as utils

if __name__ == "__main__":
    model_name,dataset_name=utils.parse_model_and_dataset("Experiment: accuracy of model for rotated vs unrotated dataset.")
else:
    dataset_name="cifar10"
    model_name=pytorch_models.AllConvolutional.__name__



verbose=False
dataset = datasets.get_dataset(dataset_name)
if verbose:
    print(f"Training with dataset {dataset_name}.")
    print(dataset.summary())

# MODEL

model, optimizer, rotated_model, rotated_optimizer = models.get_model(model_name,dataset,use_cuda)


if verbose:
    print(f"Training with model {model_name}.")
    print(model)
    print(rotated_model)

# TRAINING
pre_rotated_epochs=0
batch_size = 64
epochs,rotated_epochs=models.get_epochs(dataset.name,model_name)
config=rotation.TrainRotatedConfig(batch_size=batch_size,
                       epochs=epochs,rotated_epochs=rotated_epochs,
                       pre_rotated_epochs=pre_rotated_epochs, optimizer=optimizer,rotated_optimizer=rotated_optimizer,
                      use_cuda=use_cuda)

scores=rotation.run(config,model,rotated_model,dataset,plot_accuracy=True,save_plots=True)
rotation.print_scores(scores)

# SAVING
save_model=True
if save_model:
    rotation.save_models(dataset,model,rotated_model,scores,config)

