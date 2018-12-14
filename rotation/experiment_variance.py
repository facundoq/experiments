import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
logging.getLogger().setLevel(logging.DEBUG)

import pytorch_models
from pytorch import dataset as datasets
import torch

model_name=pytorch_models.AllConv.__name__
dataset_name="cifar10"
print(f"### Loading dataset {dataset_name} and model {model_name}.")
verbose=True

use_cuda=torch.cuda.is_available()

dataset = datasets.get_dataset(dataset_name)
if verbose:
    print(dataset.summary())

from pytorch.experiment import rotation
model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)
if verbose:
    print("### ", model)
    print("### ", rotated_model)
    print("### Scores obtained:")
    rotation.print_scores(scores)

from pytorch.experiment import variance
variance.run_and_plot_all(model,rotated_model,dataset, config, n_rotations = 16)