import os
import logging

def run_experiment(experiment, model_name, dataset_name):
    command=f"python {experiment}.py {model_name} {dataset_name}"
    logging.info(f"Running {command}")
    os.system(command)

# DATASET
import datasets

from pytorch.experiment import models

model_names=models.get_model_names()
train=False
experiments=["experiment_variance","experiment_accuracy_vs_rotation"]

for model_name in model_names:
    for dataset_name in datasets.names:
        if train:
            run_experiment("experiment_rotation",model_name,dataset_name)
        for experiment in experiments:
            run_experiment(experiment,model_name,dataset_name)
