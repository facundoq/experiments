import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import logging
import subprocess
import sys

def run_experiment(experiment, model_name, dataset_name):
    command=f"pwd && source /home/facundo/faq/exp/.env/bin/activate && python3 {experiment}.py {model_name} {dataset_name}"
    print(f"Running {command}")
    logging.info(f"Running {command}")
    subprocess.call(f'/bin/bash -c "{command}"', shell=True)
    # TODO use
    #sys.executable

# DATASET
import datasets

from pytorch.experiment import models

model_names=models.get_model_names()
train=True
experiments=["experiment_variance","experiment_accuracy_vs_rotation"]

for model_name in model_names:
    for dataset_name in datasets.names:
        if train:
            run_experiment("experiment_rotation",model_name,dataset_name)
        for experiment in experiments:
            run_experiment(experiment,model_name,dataset_name)
