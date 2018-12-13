import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
logging.getLogger().setLevel(logging.DEBUG)

verbose=True

import torch
use_cuda=torch.cuda.is_available()
from pytorch import dataset as ptd
import datasets

dataset_name="cifar10"

(x_train, y_train), (x_test, y_test), input_shape,num_classes = datasets.get_data(dataset_name)
dataset=ptd.ClassificationDataset(dataset_name,x_train,x_test,y_train,y_test,num_classes,input_shape)


from pytorch.experiment import models,rotation
model_name="SimpleConv"
model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)

print(f"### Evaluating with dataset {dataset_name} and model {model_name}.")
if verbose:
    print("### ", model)
    print("### ", rotated_model)
    print("### Scores obtained:")
    rotation.print_scores(scores)


from pytorch.experiment import variance
variance.run_and_plot_all(model,rotated_model,dataset, config, n_rotations = 16)