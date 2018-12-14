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
dataset_name="cifar10"
dataset = datasets.get_dataset(dataset_name)

verbose=False
print(f"Training with dataset {dataset_name}.")
if verbose:
    print(dataset.summary())

# MODEL
from pytorch.experiment import models,rotation
import pytorch_models
model_name=pytorch_models.AllConvolutional.__name__
model, optimizer, rotated_model, rotated_optimizer = models.get_model(model_name,dataset,use_cuda)

if verbose:
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

